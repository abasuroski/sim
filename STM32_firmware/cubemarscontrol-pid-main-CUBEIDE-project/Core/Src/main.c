/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Robot arm UART remote control (4 motors + 3 servos)
  *
  * Thin passthrough: receives MIT control commands from a PC controller
  * over UART, packs CAN frames, and forwards motor feedback back over UART.
  * All trajectory generation, PID, and gain scheduling live on the PC side.
  *
  * Controls up to 4 CubeMars motors (AK60, AK70/80, or AK40) via CAN bus,
  * and 3 MG995 servos via TIM2 PWM (end effector).
  *
  * Timing: the main loop runs every 2 ms (LOOP_HZ, paced by the DWT cycle
  * counter). Each pass sends every motor one CAN frame and, at the end of the
  * pass, one "FB" line with the replies to the PC.
  *
  * PC -> STM32 protocol (2 Mbaud): <motor_num><cmd><value>[*HH]\n
  *   A<64 hex> = all four motors' MIT commands, packed for CAN (8 bytes
  *               each, motor 1 first, mit_pack_cmd layout), applied together
  *   motor_num = 1-4 (Body: Base, Shoulder, Elbow, Linkage)
  *   motor_num = 5-7 (End Effector: Wrist1, Wrist2, Gripper)
  *   P=position (rad for motors, degrees for servos)
  *   V=velocity feedforward, K=inner MIT kp, D=inner MIT kd, T=torque offset
  *   M<pos>,<vel>,<kp>,<kd>,<tff> = all five MIT fields, applied together
  *   E=enable, R=passive feedback (0xFC poll, no torque), O=set origin
  *   0X = emergency stop (hold all motors at current position)
  *   *HH = XOR of every byte before '*', two hex digits. Required on value
  *         commands (A/P/V/K/D/T/M and servo P); optional on E/R/O/0X.
  *
  * STM32 -> PC: CAN data as hex, decoded on the PC:
  *   FB <m1> <m2> <m3> <m4> <HAL tick ms, 8 hex>\r\n   (once per 2 ms pass)
  *        each <mN> = the 16-hex feedback CAN data of that motor's latest
  *        reply this pass, or "-" if none. AK60 data has its position made
  *        relative to the software origin.
  *   C<n> <8 bytes of MIT command as packed for CAN>\r\n   (TX_ECHO: on
  *        change, at most every ECHO_MIN_INTERVAL_MS, plus a heartbeat)
  *   [UART] ..., [Mn] ..., PONG ... stay plain text.
  *
  * AK60 uses extended CAN ID:  ExtId = CAN_ID | (8 << 8)
  * AK70/80 uses standard CAN ID:  StdId = CAN_ID
  * AK40 uses standard CAN ID (same frame as AK70, different limits)
  *
  * Servo PWM: TIM2 @ 50Hz (prescaler=84-1, period=20000-1)
  *   CH1 (PA0) = Wrist 1, CH2 (PA1) = Wrist 2, CH3 (PB10) = Gripper
  *   Pulse 500 = 0 deg, Pulse 2500 = 180 deg (MG995 TowerPro)
  *
  * Hardware: STM32F446RE Nucleo + Waveshare CAN shield
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include "uart_cmd.h"
#include "mit_frame.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Private variables ---------------------------------------------------------*/
CAN_HandleTypeDef hcan1;
TIM_HandleTypeDef htim2;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

// USART2 DMA (MX_USART2_DMA_Init; not in the .ioc)
DMA_HandleTypeDef hdma_usart2_rx;
DMA_HandleTypeDef hdma_usart2_tx;

// Main loop rate. Every pass sends each motor one CAN frame.
#define LOOP_HZ  500
static uint32_t loop_period_cycles;    // SystemCoreClock / LOOP_HZ
static volatile uint32_t loop_overruns = 0;   // passes that took longer than one period

// AK60 servo-mode feedback reports position in 0.1 deg counts
#define AK60_RAD_PER_COUNT  (0.1f * 0.01745329f)

#define CAN_PACKET_MIT 8

// Feedback watchdog: if an enabled motor stops responding (or never responds
// after its enable) for this many ms, disable it and send zero-gain frames to
// prevent snap on power recovery.
#define FB_WATCHDOG_MS  2000

// Set to 1 to echo outgoing CAN command frames back to the PC over UART.
// Only for enabled motors (not zero-torque idle or soft-start frames), and
// only when the packed frame changes, at most every ECHO_MIN_INTERVAL_MS (an
// echo per 2 ms pass would cost as much UART as the feedback itself), plus
// every ECHO_HEARTBEAT_MS so the PC can tell an unchanged command from a
// silent STM32.
#define TX_ECHO  1
#define ECHO_MIN_INTERVAL_MS  20
#define ECHO_HEARTBEAT_MS     250

// Servo PWM: 50Hz, 1us resolution (prescaler=84-1, period=20000-1)
// MG995: center at 1500us, ~8.33us per degree
// Pulse clamped to 500-2500us for safety
#define SERVO_PULSE_CENTER 1500
#define SERVO_US_PER_DEG   8.333f
#define SERVO_PULSE_CLAMP_MIN  500
#define SERVO_PULSE_CLAMP_MAX  2500

// CAN handles
CAN_TxHeaderTypeDef txHeader;
CAN_RxHeaderTypeDef rxHeader;
uint8_t  txData[8];
uint8_t  rxData[8];
uint32_t txMailbox;

// UART receive: the DMA writes every byte into this circular buffer, so a
// busy CPU can no longer overrun the UART (2026-10-26: printf in the CAN ISR
// held off the receive ISR for up to 200 us, and 7-20 command bytes were lost
// at a time). HAL_UARTEx_RxEventCallback parses at line idle, half and full.
// 1024 bytes is 5 ms of the line at 2 Mbaud.
#define UART_RX_DMA_SIZE 1024
static uint8_t  uart_rx_dma[UART_RX_DMA_SIZE];
static uint16_t uart_rx_pos = 0;   // next byte of uart_rx_dma to parse

// A PONG the ISR could not send because a transmit was in progress; the
// main loop queues it next.
static char             pong_pending[32];
static volatile uint8_t pong_pending_len = 0;

// Feedback ring buffer — written by the main loop, sent by DMA. fb_tail
// advances in HAL_UART_TxCpltCallback when a DMA chunk has gone out.
#define FB_BUF_SIZE 2048
static char fb_buf[FB_BUF_SIZE];
static uint16_t fb_head = 0;
static volatile uint16_t fb_tail = 0;
static volatile uint16_t fb_tx_len = 0;   // bytes of the DMA transfer in flight, 0 = idle
static uint32_t fb_dropped_lines = 0;

// Whole lines or nothing: a line cut short by a full buffer used to run
// into the next one, and the PC could parse a value from the wrong line.
static void fb_write(const char *str, int len)
{
    uint16_t used = (uint16_t)((fb_head + FB_BUF_SIZE - fb_tail) % FB_BUF_SIZE);
    if (len <= 0 || len > FB_BUF_SIZE - 1 - used) {
        fb_dropped_lines++;
        return;
    }
    for (int i = 0; i < len; i++) {
        fb_buf[fb_head] = str[i];
        fb_head = (fb_head + 1) % FB_BUF_SIZE;
    }
}

// CAN replies, queued raw by the CAN RX ISR and formatted by the main loop.
// snprintf("%f") in the ISR took 100+ us per reply.
#define CAN_REPLY_UNKNOWN 0xFF   // motor field of a reply from an unknown CAN id
typedef struct {
    uint8_t  motor;      // 0..NUM_MOTORS-1, or CAN_REPLY_UNKNOWN
    uint8_t  rx_id;
    uint8_t  ide;
    uint8_t  data[8];
    uint32_t tick;       // HAL_GetTick() at reception
} CanReply;
#define CAN_RXQ_SIZE 64
static CanReply can_rxq[CAN_RXQ_SIZE];
static volatile uint8_t  can_rxq_head = 0;   // written by the CAN ISR
static volatile uint8_t  can_rxq_tail = 0;   // written by the main loop
static volatile uint32_t can_rxq_dropped = 0;

// Latest reply per motor since the last "FB" line.
static uint8_t  fb_frame[NUM_MOTORS][8];
static uint8_t  fb_frame_new[NUM_MOTORS];
static uint32_t fb_frame_tick = 0;
static uint32_t fb_replaced = 0;   // replies overwritten by a second one in the same pass

// UART receive error counters (incremented in HAL_UART_ErrorCallback) and
// how often the main loop may report them.
static volatile uint32_t uart_err_ore = 0;   // overrun: a byte was lost
static volatile uint32_t uart_err_fe  = 0;   // framing: bad stop bit
static volatile uint32_t uart_err_ne  = 0;   // noise on the line
static volatile uint32_t uart_err_pe  = 0;   // parity (unused at 8N1)
#define UART_REPORT_MIN_MS 100

static void count_uart_errors(uint32_t err)
{
    if (err & HAL_UART_ERROR_ORE) uart_err_ore++;
    if (err & HAL_UART_ERROR_FE)  uart_err_fe++;
    if (err & HAL_UART_ERROR_NE)  uart_err_ne++;
    if (err & HAL_UART_ERROR_PE)  uart_err_pe++;
}

#if TX_ECHO
static uint8_t  echo_last[NUM_MOTORS][8];
static uint32_t echo_last_ms[NUM_MOTORS];
static uint8_t  echo_force[NUM_MOTORS];   // echo the next frame regardless (after enable)
#endif

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_CAN1_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN PFP */
static void         MX_USART2_DMA_Init(void);
static void         send_can_frame(void);
static void         send_motor_cmd(const MotorState *m, const MotorCmd *c);
static void         send_ak70_enable(MotorState *m);
static void         send_ak70_set_origin(MotorState *m);
static void         uart_rx_start(void);
static void         delay_serviced(uint32_t ms);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */

// DWT cycle counter → microseconds (wraps at ~51 s)
static inline uint32_t get_us(void)
{
    return DWT->CYCCNT / (SystemCoreClock / 1000000U);
}

static uint32_t servo_angle_to_pulse(float angle_deg)
{
    if (angle_deg < -120.0f) angle_deg = -120.0f;
    if (angle_deg > 120.0f) angle_deg = 120.0f;
    float pulse = (float)SERVO_PULSE_CENTER + angle_deg * SERVO_US_PER_DEG;
    return (uint32_t)pulse;
}

static void update_servos(void)
{
    float *spos = uart_cmd_servo_pos();
    TIM2->CCR1 = servo_angle_to_pulse(spos[0]);
    TIM2->CCR2 = servo_angle_to_pulse(spos[1]);
    TIM2->CCR3 = servo_angle_to_pulse(spos[2]);
}

static int put_hex(char *dst, const uint8_t *src, int n)
{
    static const char digits[] = "0123456789ABCDEF";
    for (int i = 0; i < n; i++) {
        dst[2 * i]     = digits[src[i] >> 4];
        dst[2 * i + 1] = digits[src[i] & 0xF];
    }
    return 2 * n;
}

static void send_can_frame(void)
{
    uint32_t timeout = HAL_GetTick() + 5;
    while (HAL_CAN_GetTxMailboxesFreeLevel(&hcan1) == 0) {
        if (HAL_GetTick() > timeout) return;
    }
    HAL_CAN_AddTxMessage(&hcan1, &txHeader, txData, &txMailbox);
}

// AK60: extended CAN ID; AK70/80 and AK40: standard CAN ID.
static void send_motor_cmd(const MotorState *m, const MotorCmd *c)
{
    mit_pack_cmd(m->type, c, m->type == MOTOR_TYPE_AK60 ? m->pos_offset : 0.0f, txData);

    if (m->type == MOTOR_TYPE_AK60) {
        txHeader.StdId = 0;
        txHeader.ExtId = (uint32_t)m->can_id | ((uint32_t)CAN_PACKET_MIT << 8);
        txHeader.IDE   = CAN_ID_EXT;
    } else {
        txHeader.StdId = m->can_id;
        txHeader.ExtId = 0;
        txHeader.IDE   = CAN_ID_STD;
    }
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 8;
    txHeader.TransmitGlobalTime = DISABLE;
    send_can_frame();
}

#if TX_ECHO
// "C<n> <16 hex>": the command as packed for CAN, without the AK60 software
// origin so the PC can compare it with what it sent.
static void log_tx_cmd(uint8_t motor_idx, const MotorState *m, const MotorCmd *c)
{
    uint8_t frame[8];
    mit_pack_cmd(m->type, c, 0.0f, frame);
    uint32_t now = HAL_GetTick();
    uint32_t since = now - echo_last_ms[motor_idx];
    if (!echo_force[motor_idx]) {
        uint8_t same = memcmp(frame, echo_last[motor_idx], sizeof(frame)) == 0;
        // A change waits for the next allowed slot; it is not dropped,
        // because the loop calls this again every pass.
        if (same ? since < ECHO_HEARTBEAT_MS : since < ECHO_MIN_INTERVAL_MS)
            return;
    }
    echo_force[motor_idx] = 0;
    memcpy(echo_last[motor_idx], frame, sizeof(frame));
    echo_last_ms[motor_idx] = now;

    char line[24];
    int len = 0;
    line[len++] = 'C';
    line[len++] = (char)('1' + motor_idx);
    line[len++] = ' ';
    len += put_hex(&line[len], frame, 8);
    line[len++] = '\r';
    line[len++] = '\n';
    fb_write(line, len);
}
#endif

// AK70/80 enable: 0xFF x7 + 0xFC
static void send_ak70_enable(MotorState *m)
{
    txData[0] = 0xFF;
    txData[1] = 0xFF;
    txData[2] = 0xFF;
    txData[3] = 0xFF;
    txData[4] = 0xFF;
    txData[5] = 0xFF;
    txData[6] = 0xFF;
    txData[7] = 0xFC;

    txHeader.StdId              = m->can_id;
    txHeader.ExtId              = 0;
    txHeader.IDE                = CAN_ID_STD;
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 8;
    txHeader.TransmitGlobalTime = DISABLE;
    send_can_frame();
}

// AK70/80/AK40 set origin: 0xFF x7 + 0xFE
static void send_ak70_set_origin(MotorState *m)
{
    txData[0] = 0xFF;
    txData[1] = 0xFF;
    txData[2] = 0xFF;
    txData[3] = 0xFF;
    txData[4] = 0xFF;
    txData[5] = 0xFF;
    txData[6] = 0xFF;
    txData[7] = 0xFE;

    txHeader.StdId              = m->can_id;
    txHeader.ExtId              = 0;
    txHeader.IDE                = CAN_ID_STD;
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 8;
    txHeader.TransmitGlobalTime = DISABLE;
    send_can_frame();
}

// AK60 feedback (servo-mode format): int16 pos (0.1 deg), int16 speed
// (10 eRPM), int16 current (10 mA), int8 temp, int8 error. The position bytes
// are rewritten relative to the software origin before they are forwarded.
static void decode_ak60_reply(MotorState *m, uint8_t *data, uint32_t ts)
{
    int16_t pos_int = (int16_t)((data[0] << 8) | data[1]);
    int32_t rel = (int32_t)pos_int - m->pos_offset_counts;
    if (rel > INT16_MAX) rel = INT16_MAX;
    if (rel < INT16_MIN) rel = INT16_MIN;

    m->fb_raw_count = pos_int;
    m->fb_pos = (float)rel * AK60_RAD_PER_COUNT;
    m->fb_received = 1;
    m->fb_tick = ts;

    data[0] = (uint8_t)((uint16_t)rel >> 8);
    data[1] = (uint8_t)((uint16_t)rel & 0xFF);
}

// AK70/80 and AK40 feedback (MIT format): id, 16-bit pos, 12-bit vel,
// 12-bit torque, temp + 40, error. Only the position is needed here.
static void decode_mit_reply(MotorState *m, const uint8_t *data, uint32_t ts)
{
    int p_int = (data[1] << 8) | data[2];
    float p_min = (m->type == MOTOR_TYPE_AK40) ? AK40_P_MIN : AK70_P_MIN;
    float p_max = (m->type == MOTOR_TYPE_AK40) ? AK40_P_MAX : AK70_P_MAX;
    m->fb_pos = mit_uint_to_float(p_int, p_min, p_max, 16);
    m->fb_received = 1;
    m->fb_tick = ts;
}

static void can_rxq_push(uint8_t motor, uint8_t rx_id, uint8_t ide, const uint8_t *data, uint32_t ts)
{
    uint8_t next = (uint8_t)((can_rxq_head + 1) % CAN_RXQ_SIZE);
    if (next == can_rxq_tail) {
        can_rxq_dropped++;
        return;
    }
    CanReply *r = &can_rxq[can_rxq_head];
    r->motor = motor;
    r->rx_id = rx_id;
    r->ide   = ide;
    memcpy(r->data, data, 8);
    r->tick  = ts;
    can_rxq_head = next;
}

// Identify which motor sent feedback based on CAN ID. Kept short: decode the
// position the watchdog and enable sequence need, queue the raw bytes.
void HAL_CAN_RxFifo0MsgPendingCallback(CAN_HandleTypeDef *hcan)
{
    if (HAL_CAN_GetRxMessage(hcan, CAN_RX_FIFO0, &rxHeader, rxData) != HAL_OK)
        return;

    // Try multiple ways to identify source motor:
    //  - Extended frame (AK60): motor ID in lower byte of ExtId
    //  - Standard frame (AK70): motor ID in rxData[0]
    uint8_t rx_id;
    if (rxHeader.IDE == CAN_ID_EXT)
        rx_id = (uint8_t)(rxHeader.ExtId & 0xFF);
    else
        rx_id = rxData[0];
    uint8_t ide = (uint8_t)rxHeader.IDE;
    uint32_t ts = HAL_GetTick();

    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        MotorState *m = uart_cmd_motor(i);
        if (m->can_id == rx_id) {
            if (m->type == MOTOR_TYPE_AK60)
                decode_ak60_reply(m, rxData, ts);
            else
                decode_mit_reply(m, rxData, ts);
            can_rxq_push(i, rx_id, ide, rxData, ts);
            HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
            return;
        }
    }

    // No match — queue for the debug line
    can_rxq_push(CAN_REPLY_UNKNOWN, rx_id, ide, rxData, ts);
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
}

// Move queued CAN replies into fb_frame for the next "FB" line (main loop).
// Replies from unknown CAN ids become their own debug line.
static void drain_can_queue(void)
{
    while (can_rxq_tail != can_rxq_head) {
        const CanReply *r = &can_rxq[can_rxq_tail];
        if (r->motor < NUM_MOTORS) {
            if (fb_frame_new[r->motor]) fb_replaced++;
            memcpy(fb_frame[r->motor], r->data, 8);
            fb_frame_new[r->motor] = 1;
            fb_frame_tick = r->tick;
        } else {
            char line[48];
            int len = snprintf(line, sizeof(line), "[M0] unknown CAN id=%d ide=%lu\r\n",
                               (int)r->rx_id, (unsigned long)r->ide);
            fb_write(line, len);
        }
        can_rxq_tail = (uint8_t)((can_rxq_tail + 1) % CAN_RXQ_SIZE);
    }
}

// "FB <m1> <m2> <m3> <m4> <tick>": this pass's replies, one line for all
// four motors (four separate lines cost a third more UART at 500 Hz).
static void emit_feedback_line(void)
{
    drain_can_queue();
    uint8_t any = 0;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) any |= fb_frame_new[i];
    if (!any) return;

    char line[96];
    int len = 0;
    line[len++] = 'F';
    line[len++] = 'B';
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        line[len++] = ' ';
        if (fb_frame_new[i])
            len += put_hex(&line[len], fb_frame[i], 8);
        else
            line[len++] = '-';
        fb_frame_new[i] = 0;
    }
    uint8_t ts[4] = {
        (uint8_t)(fb_frame_tick >> 24), (uint8_t)(fb_frame_tick >> 16),
        (uint8_t)(fb_frame_tick >> 8),  (uint8_t)fb_frame_tick,
    };
    line[len++] = ' ';
    len += put_hex(&line[len], ts, 4);
    line[len++] = '\r';
    line[len++] = '\n';
    fb_write(line, len);
}

// One received byte (UART IDLE / DMA ISR context).
static void uart_rx_byte(uint8_t byte)
{
    // Detect "PING\n" and respond immediately with a timestamped PONG.
    // Responding in the ISR (not the ring buffer) keeps the round-trip
    // time as short as possible for accurate clock-offset estimation.
    // "PING\n" is harmless to uart_cmd_feed: 'P'-'1' is out of motor range.
    static char  ping_buf[5];
    static uint8_t ping_idx = 0;
    if (byte == '\n' || byte == '\r') {
        ping_buf[ping_idx] = '\0';
        if (ping_idx == 4 &&
            ping_buf[0] == 'P' && ping_buf[1] == 'I' &&
            ping_buf[2] == 'N' && ping_buf[3] == 'G') {
            char pong[32];
            int plen = snprintf(pong, sizeof(pong), "PONG t=%luus\r\n",
                                (unsigned long)get_us());
            // HAL_BUSY while a feedback DMA transfer is running: the main
            // loop queues it next.
            if (HAL_UART_Transmit(&huart2, (uint8_t *)pong, (uint16_t)plen, 10) != HAL_OK) {
                memcpy(pong_pending, pong, (size_t)plen);
                pong_pending_len = (uint8_t)plen;
            }
        }
        ping_idx = 0;
    } else if (ping_idx < 4) {
        ping_buf[ping_idx++] = (char)byte;
    }

    uart_cmd_feed(byte);
}

// Parse every byte the DMA has written before index `end` (1..UART_RX_DMA_SIZE).
static void uart_rx_parse_to(uint16_t end)
{
    if (end >= UART_RX_DMA_SIZE) end = 0;   // full buffer: parse up to the wrap
    while (uart_rx_pos != end) {
        uart_rx_byte(uart_rx_dma[uart_rx_pos]);
        uart_rx_pos = (uint16_t)((uart_rx_pos + 1) % UART_RX_DMA_SIZE);
    }
}

static void uart_rx_start(void)
{
    uart_rx_pos = 0;
    HAL_UARTEx_ReceiveToIdle_DMA(&huart2, uart_rx_dma, UART_RX_DMA_SIZE);
}

// Line idle, half buffer or full buffer: Size is the DMA write position.
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
    if (huart->Instance != USART2) return;
    uart_rx_parse_to(Size);
}

// A feedback DMA chunk has gone out.
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART2) return;
    fb_tail = (uint16_t)((fb_tail + fb_tx_len) % FB_BUF_SIZE);
    fb_tx_len = 0;
}

// In DMA mode the HAL treats every receive error as blocking: it aborts the
// RX DMA and ends here. The byte the error hit cannot be identified, so the
// line in progress is marked corrupt before and after the bytes still
// unparsed; value commands also need a valid checksum. Then reception
// restarts. A TX DMA error also ends here, with receive still running: then
// only the feedback chunk is retried.
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART2) return;
    uint32_t err = huart->ErrorCode;
    count_uart_errors(err);

    if (huart->gState == HAL_UART_STATE_READY)
        fb_tx_len = 0;   // no transmit running: resend from fb_tail

    if (huart->RxState != HAL_UART_STATE_READY)
        return;          // receive was not stopped
    uart_cmd_mark_line_corrupt();
    if (huart->hdmarx != NULL)
        uart_rx_parse_to((uint16_t)(UART_RX_DMA_SIZE - __HAL_DMA_GET_COUNTER(huart->hdmarx)));
    uart_cmd_mark_line_corrupt();

    if (__HAL_UART_GET_FLAG(huart, UART_FLAG_ORE))
        __HAL_UART_CLEAR_OREFLAG(huart);
    uart_rx_start();
}

// Report UART receive errors, rejected command lines, dropped feedback and
// loop overruns to the PC, only when something changed and at most every
// UART_REPORT_MIN_MS.
static void report_uart_errors(void)
{
    static uint32_t last_ore, last_fe, last_ne, last_pe, last_rej, last_fb_drop, last_can_drop;
    static uint32_t last_replaced, last_overruns;
    static uint32_t last_report_ms;
    uint32_t now = HAL_GetTick();
    if (now - last_report_ms < UART_REPORT_MIN_MS) return;

    uint32_t ore = uart_err_ore, fe = uart_err_fe, ne = uart_err_ne, pe = uart_err_pe;
    uint32_t rej = uart_cmd_rejected_count();
    uint32_t fb_drop = fb_dropped_lines, can_drop = can_rxq_dropped;
    uint32_t replaced = fb_replaced, overruns = loop_overruns;
    if (ore == last_ore && fe == last_fe && ne == last_ne && pe == last_pe && rej == last_rej &&
        fb_drop == last_fb_drop && can_drop == last_can_drop &&
        replaced == last_replaced && overruns == last_overruns)
        return;
    last_ore = ore; last_fe = fe; last_ne = ne; last_pe = pe; last_rej = rej;
    last_fb_drop = fb_drop; last_can_drop = can_drop;
    last_replaced = replaced; last_overruns = overruns;
    last_report_ms = now;

    char line[200];
    int len = snprintf(line, sizeof(line),
        "[UART] rx errors ore=%lu fe=%lu ne=%lu pe=%lu rejected=%lu fb_drop=%lu can_drop=%lu "
        "fb_replaced=%lu loop_over=%lu t=%lums\r\n",
        (unsigned long)ore, (unsigned long)fe, (unsigned long)ne,
        (unsigned long)pe, (unsigned long)rej, (unsigned long)fb_drop,
        (unsigned long)can_drop, (unsigned long)replaced,
        (unsigned long)overruns, (unsigned long)now);
    fb_write(line, len);

    // The latest rejected line, with non-printable bytes as \xNN, so the
    // exact damage is visible in the PC run log.
    char raw[UART_CMD_LINE_MAX];
    if (uart_cmd_take_rejected(raw, sizeof(raw))) {
        len = snprintf(line, sizeof(line), "[UART] rejected \"");
        for (int i = 0; raw[i] != '\0' && len < (int)sizeof(line) - 8; i++) {
            unsigned char c = (unsigned char)raw[i];
            if (c >= 0x20 && c < 0x7F && c != '"' && c != '\\')
                line[len++] = (char)c;
            else
                len += snprintf(&line[len], sizeof(line) - len, "\\x%02X", c);
        }
        len += snprintf(&line[len], sizeof(line) - len, "\"\r\n");
        fb_write(line, len);
    }
}

// Start a DMA transfer of the next contiguous part of fb_buf, unless one is
// still running (main loop only). Never blocks.
static void drain_fb(void)
{
    if (pong_pending_len) {
        char pong[sizeof(pong_pending)];
        uint8_t plen;
        __disable_irq();
        plen = pong_pending_len;
        memcpy(pong, pong_pending, plen);
        pong_pending_len = 0;
        __enable_irq();
        fb_write(pong, plen);
    }

    if (fb_tx_len != 0 || huart2.gState != HAL_UART_STATE_READY) return;
    uint16_t tail = fb_tail;
    if (tail == fb_head) return;
    uint16_t chunk = (fb_head > tail) ? (uint16_t)(fb_head - tail) : (uint16_t)(FB_BUF_SIZE - tail);
    fb_tx_len = chunk;
    if (HAL_UART_Transmit_DMA(&huart2, (uint8_t *)&fb_buf[tail], chunk) != HAL_OK)
        fb_tx_len = 0;   // try again on the next call
}

static void service_io(void)
{
    drain_can_queue();
    report_uart_errors();
    drain_fb();
}

// HAL_Delay that keeps forwarding feedback (startup and set-origin waits).
static void delay_serviced(uint32_t ms)
{
    uint32_t start = HAL_GetTick();
    do {
        service_io();
        emit_feedback_line();
    } while (HAL_GetTick() - start <= ms);
}

// Service I/O until one loop period has passed since `start` (a DWT count).
static void wait_period(uint32_t start)
{
    do {
        service_io();
    } while ((uint32_t)(DWT->CYCCNT - start) < loop_period_cycles);
}

// Freeze all enabled motors at their last known feedback position.
static void do_e_stop(void)
{
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        MotorState *m = uart_cmd_motor(i);
        if (m->enabled) {
            if (m->fb_received)
                m->pos = m->fb_pos;
            m->tff = 0.0f;
        }
    }
}

// B1 button (PC13) — physical emergency stop
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == B1_Pin)
        do_e_stop();
}

static void clamp_tff(const MotorState *m, MotorCmd *c)
{
    float t_max;
    if (m->type == MOTOR_TYPE_AK60)
        t_max = AK60_T_MAX;
    else if (m->type == MOTOR_TYPE_AK40)
        t_max = AK40_T_MAX;
    else
        t_max = AK70_T_MAX;
    if (c->tff > t_max) c->tff = t_max;
    if (c->tff < -t_max) c->tff = -t_max;
}

// One main-loop round of frames for every motor except `skip`, used while
// `skip` is being enabled. The enable sequence blocks for ~1.3 s per motor,
// and an AK70/AK40 only replies when it receives a frame, so without this an
// already-enabled motor went silent and the feedback watchdog disabled it
// (M2 on 2026-10-26 and on 2026-10-27, while M3 and M4 were being enabled).
static void service_other_motors(uint8_t skip)
{
    if (uart_cmd_e_stop_requested()) {
        do_e_stop();
        uart_cmd_clear_e_stop();
    }
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        if (i == skip) continue;
        MotorState *m = uart_cmd_motor(i);
        if (m->fb_only) {
            if (m->type != MOTOR_TYPE_AK60)
                send_ak70_enable(m);  // 0xFC poll, as in the main loop
            continue;
        }
        MotorCmd c;
        uart_cmd_read(i, &c);
        if (!m->enabled) {
            c.kp = 0.0f; c.kd = 0.0f; c.tff = 0.0f;
            send_motor_cmd(m, &c);
            continue;
        }
        clamp_tff(m, &c);
        send_motor_cmd(m, &c);
#if TX_ECHO
        log_tx_cmd(i, m, &c);
#endif
    }
}

// The enable sequence's waits: whole loop periods in which the other motors
// keep their 500 Hz frames and the PC its "FB" lines.
static void enable_delay(uint8_t enabling, uint32_t ms)
{
    uint32_t periods = (ms * LOOP_HZ + 999) / 1000;
    for (uint32_t p = 0; p < periods; p++) {
        uint32_t start = DWT->CYCCNT;
        service_other_motors(enabling);
        wait_period(start);
        emit_feedback_line();
    }
}

// Enable with soft-start. Sends copies of the command (uart_cmd_read), so
// K/D lines that arrive from the PC meanwhile are kept, not overwritten.
// Every wait goes through enable_delay so the other motors keep receiving
// frames.
static void enable_motor(uint8_t motor_idx, MotorState *m)
{
    // AK70/AK40 need MIT mode entry frame
    if (m->type != MOTOR_TYPE_AK60) {
        for (int j = 0; j < 10; j++) {
            send_ak70_enable(m);
            enable_delay(motor_idx, 10);
        }
        enable_delay(motor_idx, 100);
    }

    m->fb_received = 0;
    m->tff = 0.0f;
    MotorCmd c;

    // Probe with zero gain for all motor types so the ramp
    // starts at the actual position instead of 0.
    uint32_t probe_end = HAL_GetTick() + 300;
    while (!m->fb_received && HAL_GetTick() < probe_end) {
        uart_cmd_read(motor_idx, &c);
        c.kp  = 0.0f;
        c.kd  = 0.0f;
        c.tff = 0.0f;
        send_motor_cmd(m, &c);
        enable_delay(motor_idx, 10);
    }
    if (m->fb_received)
        m->pos = m->fb_pos;

    for (int j = 1; j <= 50; j++) {
        if (m->fb_received && j <= 5)
            m->pos = m->fb_pos;

        uart_cmd_read(motor_idx, &c);
        c.kp = c.kp * (float)j / 50.0f;
        c.kd = c.kd * (float)j / 50.0f;
        send_motor_cmd(m, &c);

        enable_delay(motor_idx, 20);
    }

    // The watchdog counts from here, so a motor that never answered is
    // disabled after FB_WATCHDOG_MS instead of staying enabled forever.
    m->fb_tick = HAL_GetTick();
    m->enable_pending = 0;
    m->enabled = 1;
#if TX_ECHO
    echo_force[motor_idx] = 1;
#endif
}

// USART2 DMA, set up here rather than in CubeMX-generated code:
//   RX: DMA1 Stream 5, Channel 4, circular.
//   TX: DMA1 Stream 6, Channel 4, normal (one transfer per fb_buf chunk).
static void MX_USART2_DMA_Init(void)
{
    __HAL_RCC_DMA1_CLK_ENABLE();

    hdma_usart2_rx.Instance                 = DMA1_Stream5;
    hdma_usart2_rx.Init.Channel             = DMA_CHANNEL_4;
    hdma_usart2_rx.Init.Direction           = DMA_PERIPH_TO_MEMORY;
    hdma_usart2_rx.Init.PeriphInc           = DMA_PINC_DISABLE;
    hdma_usart2_rx.Init.MemInc              = DMA_MINC_ENABLE;
    hdma_usart2_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart2_rx.Init.MemDataAlignment    = DMA_MDATAALIGN_BYTE;
    hdma_usart2_rx.Init.Mode                = DMA_CIRCULAR;
    hdma_usart2_rx.Init.Priority            = DMA_PRIORITY_HIGH;
    hdma_usart2_rx.Init.FIFOMode            = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_usart2_rx) != HAL_OK) Error_Handler();
    __HAL_LINKDMA(&huart2, hdmarx, hdma_usart2_rx);

    hdma_usart2_tx.Instance                 = DMA1_Stream6;
    hdma_usart2_tx.Init.Channel             = DMA_CHANNEL_4;
    hdma_usart2_tx.Init.Direction           = DMA_MEMORY_TO_PERIPH;
    hdma_usart2_tx.Init.PeriphInc           = DMA_PINC_DISABLE;
    hdma_usart2_tx.Init.MemInc              = DMA_MINC_ENABLE;
    hdma_usart2_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart2_tx.Init.MemDataAlignment    = DMA_MDATAALIGN_BYTE;
    hdma_usart2_tx.Init.Mode                = DMA_NORMAL;
    hdma_usart2_tx.Init.Priority            = DMA_PRIORITY_MEDIUM;
    hdma_usart2_tx.Init.FIFOMode            = DMA_FIFOMODE_DISABLE;
    if (HAL_DMA_Init(&hdma_usart2_tx) != HAL_OK) Error_Handler();
    __HAL_LINKDMA(&huart2, hdmatx, hdma_usart2_tx);

    // RX at the USART2 level: both call HAL_UARTEx_RxEventCallback, which
    // must not interrupt itself. TX completion is not urgent.
    HAL_NVIC_SetPriority(DMA1_Stream5_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Stream5_IRQn);
    HAL_NVIC_SetPriority(DMA1_Stream6_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA1_Stream6_IRQn);
}

/* USER CODE END 0 */

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_TIM2_Init();
    MX_CAN1_Init();
    MX_USART2_UART_Init();

    /* USER CODE BEGIN 2 */

    MX_USART2_DMA_Init();

    // Enable DWT cycle counter for microsecond timestamps and loop pacing
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT       = 0;
    DWT->CTRL        |= DWT_CTRL_CYCCNTENA_Msk;
    loop_period_cycles = SystemCoreClock / LOOP_HZ;

    HAL_CAN_Start(&hcan1);
    HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING);
    // Below USART2/DMA (0). Needs NVIC_PRIORITYGROUP_4 (HAL_MspInit): under
    // GROUP_0 the preempt level is masked off and UART cannot interrupt CAN.
    HAL_NVIC_SetPriority(CAN1_RX0_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(CAN1_RX0_IRQn);

    // Servos already started by MX_TIM2_Init (direct register config)
    update_servos();

    uart_cmd_init();
    HAL_NVIC_SetPriority(USART2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
    uart_rx_start();

    // Enable B1 button (PC13) as emergency stop
    HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

    HAL_Delay(100);

    // Send zero-torque to all motors on startup to suppress power-on snap.
    // Motors that retained MIT mode from a previous session will receive kp=0,
    // kd=0, tff=0 and stay limp instead of snapping to their stale setpoint.
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        MotorState *m = uart_cmd_motor(i);
        MotorCmd c;
        uart_cmd_read(i, &c);
        c.kp  = 0.0f;
        c.kd  = 0.0f;
        c.tff = 0.0f;
        for (int j = 0; j < 3; j++) {
            send_motor_cmd(m, &c);
            delay_serviced(10);
        }
    }

    /* USER CODE END 2 */

    /* USER CODE BEGIN WHILE */
    while (1)
    {
        uint32_t loop_start = DWT->CYCCNT;

        // UART-triggered emergency stop
        if (uart_cmd_e_stop_requested()) {
            do_e_stop();
            uart_cmd_clear_e_stop();
        }

        for (uint8_t i = 0; i < NUM_MOTORS; i++)
        {
            MotorState *m = uart_cmd_motor(i);

            // Handle set origin
            if (m->set_origin_pending) {
                if (m->type == MOTOR_TYPE_AK60) {
                    // Software-only origin: offset translates user coords to MIT coords
                    m->pos_offset_counts = m->fb_raw_count;
                    m->pos_offset = (float)m->pos_offset_counts * AK60_RAD_PER_COUNT;
                } else {
                    send_ak70_set_origin(m);
                }
                delay_serviced(10);
                m->pos = 0.0f;
                m->fb_pos = 0.0f;
                m->set_origin_pending = 0;
            }

            // Handle pending enable with soft-start
            if (m->enable_pending)
                enable_motor(i, m);

            // Passive feedback mode: poll state via 0xFC, no torque applied
            if (m->fb_only) {
                if (m->type != MOTOR_TYPE_AK60)
                    send_ak70_enable(m);  // 0xFC — triggers state response, no control
                continue;
            }

            MotorCmd c;
            uart_cmd_read(i, &c);

            // Non-enabled motors get a continuous zero-torque packet so that
            // if they power on after the STM32 they still snap to nothing.
            if (!m->enabled) {
                c.kp = 0.0f; c.kd = 0.0f; c.tff = 0.0f;
                send_motor_cmd(m, &c);
                continue;
            }

            // Feedback watchdog: if this motor lost power, stopped responding
            // or never responded, disable it so it doesn't snap on recovery.
            if ((HAL_GetTick() - m->fb_tick) > FB_WATCHDOG_MS) {
                m->enabled = 0;
                m->tff = 0.0f;
                char wdg[72];
                int wlen = snprintf(wdg, sizeof(wdg),
                    "[M%d] %s — disabled (re-enable required)\r\n",
                    i + 1, m->fb_received ? "feedback lost" : "no feedback since enable");
                fb_write(wdg, wlen);
                continue;
            }

            // Clamp torque to motor limits
            clamp_tff(m, &c);
            send_motor_cmd(m, &c);

#if TX_ECHO
            log_tx_cmd(i, m, &c);
#endif
        }

        // Update servo PWM outputs
        update_servos();

        // Wait out the rest of the 2 ms period, forwarding feedback
        // meanwhile; the replies to this pass's frames arrive within it.
        if ((uint32_t)(DWT->CYCCNT - loop_start) >= loop_period_cycles)
            loop_overruns++;
        wait_period(loop_start);
        emit_feedback_line();
    }
    /* USER CODE END WHILE */
}

/* -------------------------------------------------------------------------- */
/*  Peripheral init                                                           */
/* -------------------------------------------------------------------------- */

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    // Scale 1: full voltage regulation — better stability for active control vs low-power Scale 3
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    // HSE via ST-Link MCO on PH0 (bypass mode, 8 MHz).
    // SYSCLK = 8 MHz / 4 * 168 / 4 = 84 MHz — same frequency as before, but crystal-accurate.
    // If the board hangs here, check that SB54/SB55 solder bridges are closed on the Nucleo.
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState       = RCC_HSE_BYPASS;
    RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM       = 4;    // 8 MHz / 4 = 2 MHz VCO input
    RCC_OscInitStruct.PLL.PLLN       = 168;  // 2 MHz * 168 = 336 MHz VCO
    RCC_OscInitStruct.PLL.PLLP       = RCC_PLLP_DIV4;  // 336 / 4 = 84 MHz SYSCLK
    RCC_OscInitStruct.PLL.PLLQ       = 7;    // 336 / 7 = 48 MHz (USB)
    RCC_OscInitStruct.PLL.PLLR       = 2;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

    RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                     | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) Error_Handler();
}

static void MX_CAN1_Init(void)
{
    hcan1.Instance              = CAN1;
    hcan1.Init.Prescaler        = 3;
    hcan1.Init.Mode             = CAN_MODE_NORMAL;
    hcan1.Init.SyncJumpWidth    = CAN_SJW_1TQ;
    hcan1.Init.TimeSeg1         = CAN_BS1_11TQ;
    hcan1.Init.TimeSeg2         = CAN_BS2_2TQ;
    hcan1.Init.TimeTriggeredMode    = DISABLE;
    hcan1.Init.AutoBusOff           = ENABLE;
    hcan1.Init.AutoWakeUp           = DISABLE;
    hcan1.Init.AutoRetransmission   = DISABLE;
    hcan1.Init.ReceiveFifoLocked    = DISABLE;
    hcan1.Init.TransmitFifoPriority = DISABLE;
    if (HAL_CAN_Init(&hcan1) != HAL_OK) Error_Handler();

    CAN_FilterTypeDef filter;
    filter.FilterActivation     = ENABLE;
    filter.FilterBank           = 0;
    filter.FilterFIFOAssignment = CAN_FILTER_FIFO0;
    filter.FilterIdHigh         = 0x0000;
    filter.FilterIdLow          = 0x0000;
    filter.FilterMaskIdHigh     = 0x0000;
    filter.FilterMaskIdLow      = 0x0000;
    filter.FilterMode           = CAN_FILTERMODE_IDMASK;
    filter.FilterScale          = CAN_FILTERSCALE_32BIT;
    HAL_CAN_ConfigFilter(&hcan1, &filter);
}

static void MX_TIM2_Init(void)
{
    // Direct register setup — bypasses HAL state machine issues

    // 1. Enable TIM2 clock
    RCC->APB1ENR |= RCC_APB1ENR_TIM2EN;
    __DSB();  // ensure clock is active before accessing registers

    // 2. Configure PA0 (TIM2_CH1) and PA1 (TIM2_CH2) as AF1
    //    MODER = 10 (alternate function), AFRL = 0001 (AF1 = TIM2)
    GPIOA->MODER   &= ~(GPIO_MODER_MODER0 | GPIO_MODER_MODER1);
    GPIOA->MODER   |=  (2U << (0*2)) | (2U << (1*2));  // AF mode for PA0, PA1
    GPIOA->OSPEEDR |=  (1U << (0*2)) | (1U << (1*2));  // Medium speed
    GPIOA->OTYPER  &= ~(GPIO_OTYPER_OT0 | GPIO_OTYPER_OT1);  // Push-pull
    GPIOA->PUPDR   &= ~(GPIO_PUPDR_PUPD0 | GPIO_PUPDR_PUPD1);  // No pull
    GPIOA->AFR[0]  &= ~(0xFU << (0*4)) & ~(0xFU << (1*4));
    GPIOA->AFR[0]  |=  (1U << (0*4)) | (1U << (1*4));  // AF1 for PA0, PA1

    // 3. Configure PB10 (TIM2_CH3) as AF1
    GPIOB->MODER   &= ~(GPIO_MODER_MODER10);
    GPIOB->MODER   |=  (2U << (10*2));  // AF mode
    GPIOB->OSPEEDR |=  (1U << (10*2));  // Medium speed
    GPIOB->OTYPER  &= ~(GPIO_OTYPER_OT10);  // Push-pull
    GPIOB->PUPDR   &= ~(GPIO_PUPDR_PUPD10);  // No pull
    GPIOB->AFR[1]  &= ~(0xFU << ((10-8)*4));
    GPIOB->AFR[1]  |=  (1U << ((10-8)*4));  // AF1 for PB10

    // 4. Configure TIM2: 50Hz PWM, 1us resolution
    TIM2->PSC  = 84 - 1;     // 84MHz / 84 = 1MHz
    TIM2->ARR  = 20000 - 1;  // 1MHz / 20000 = 50Hz
    TIM2->CCR1 = 1500;       // 1.5ms = 90 deg
    TIM2->CCR2 = 1500;
    TIM2->CCR3 = 1500;

    // PWM mode 1 on CH1, CH2 (CCMR1), CH3 (CCMR2)
    TIM2->CCMR1 = (6U << TIM_CCMR1_OC1M_Pos) | TIM_CCMR1_OC1PE
                 | (6U << TIM_CCMR1_OC2M_Pos) | TIM_CCMR1_OC2PE;
    TIM2->CCMR2 = (6U << TIM_CCMR2_OC3M_Pos) | TIM_CCMR2_OC3PE;

    // Enable CH1, CH2, CH3 outputs
    TIM2->CCER = TIM_CCER_CC1E | TIM_CCER_CC2E | TIM_CCER_CC3E;

    // Generate update event to load prescaler/ARR, then start counter
    TIM2->EGR = TIM_EGR_UG;
    TIM2->CR1 = TIM_CR1_CEN;
}

static void MX_USART2_UART_Init(void)
{
    huart2.Instance          = USART2;
    huart2.Init.BaudRate     = 2000000;   // 42 MHz / 16 / 2 M = 1.3125: exact BRR, 0% error
    huart2.Init.WordLength   = UART_WORDLENGTH_8B;
    huart2.Init.StopBits     = UART_STOPBITS_1;
    huart2.Init.Parity       = UART_PARITY_NONE;
    huart2.Init.Mode         = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}

static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

    GPIO_InitStruct.Pin  = B1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

    GPIO_InitStruct.Pin   = LD2_Pin;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {}
#endif
