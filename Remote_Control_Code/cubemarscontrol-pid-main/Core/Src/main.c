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
  * Protocol: <motor_num><cmd><value>\n
  *   motor_num = 1-4 (Body: Base, Shoulder, Elbow, Linkage)
  *   motor_num = 5-7 (End Effector: Wrist1, Wrist2, Gripper)
  *   P=position (rad for motors, degrees for servos)
  *   V=velocity feedforward, K=inner MIT kp, D=inner MIT kd, T=torque offset
  *   E=enable, R=passive feedback (0xFC poll, no torque)
  *   0X = emergency stop (hold all motors at current position)
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
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Private variables ---------------------------------------------------------*/
CAN_HandleTypeDef hcan1;
TIM_HandleTypeDef htim2;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

// AK60 parameter limits
#define AK60_P_MIN    -12.56f
#define AK60_P_MAX     12.56f
#define AK60_V_MIN    -60.0f
#define AK60_V_MAX     60.0f
#define AK60_T_MIN    -12.0f
#define AK60_T_MAX     12.0f

// AK70/80 parameter limits
#define AK70_P_MIN    -12.5f
#define AK70_P_MAX     12.5f
#define AK70_V_MIN    -30.0f
#define AK70_V_MAX     30.0f
#define AK70_T_MIN    -18.0f
#define AK70_T_MAX     18.0f

// AK40 parameter limits
#define AK40_P_MIN    -12.5f
#define AK40_P_MAX     12.5f
#define AK40_V_MIN    -45.5f
#define AK40_V_MAX     45.5f
#define AK40_T_MIN    -5.0f
#define AK40_T_MAX     5.0f

// Shared limits
#define KP_MIN    0.0f
#define KP_MAX    500.0f
#define KD_MIN    0.0f
#define KD_MAX    5.0f

#define CAN_PACKET_MIT 8

// Feedback watchdog: if an enabled motor stops responding for this many ms,
// disable it and send zero-gain frames to prevent snap on power recovery.
#define FB_WATCHDOG_MS  2000

// Set to 1 to echo every outgoing CAN command frame back to the PC over UART.
// Only fires for enabled motors (not zero-torque idle or soft-start frames).
// Toggle off in production to save ~30 KB/s of UART bandwidth.
#define TX_ECHO  1

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

uint8_t uart_rx_byte;

// Feedback ring buffer — ISR writes, main loop transmits
#define FB_BUF_SIZE 1024
static char fb_buf[FB_BUF_SIZE];
static volatile uint16_t fb_head = 0;
static volatile uint16_t fb_tail = 0;

static void fb_write(const char *str, int len)
{
    for (int i = 0; i < len; i++) {
        uint16_t next = (fb_head + 1) % FB_BUF_SIZE;
        if (next == fb_tail) break;  // full, drop
        fb_buf[fb_head] = str[i];
        fb_head = next;
    }
}

// fb_write from the main loop. The CAN RX interrupt also writes fb_buf, and
// without this the two interleaved into lines like "[M1[TX M1] pos=...".
static void fb_write_main(const char *str, int len)
{
    HAL_NVIC_DisableIRQ(CAN1_RX0_IRQn);
    fb_write(str, len);
    HAL_NVIC_EnableIRQ(CAN1_RX0_IRQn);
}

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

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_CAN1_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN PFP */
static unsigned int float_to_uint(float x, float x_min, float x_max, unsigned int bits);
static float        uint_to_float(int x_int, float x_min, float x_max, int bits);
static void         send_can_frame(void);
static void         send_ak60_cmd(MotorState *m);
static void         send_ak70_cmd(MotorState *m);
static void         send_ak40_cmd(MotorState *m);
static void         send_ak70_enable(MotorState *m);
static void         send_ak60_set_origin(MotorState *m);
static void         send_ak70_set_origin(MotorState *m);
static void         unpack_ak60_reply(uint8_t motor_idx);
static void         unpack_ak70_reply(uint8_t motor_idx);
static void         unpack_ak40_reply(uint8_t motor_idx);
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

static unsigned int float_to_uint(float x, float x_min, float x_max, unsigned int bits)
{
    float span = x_max - x_min;
    if (x < x_min) x = x_min;
    else if (x > x_max) x = x_max;
    return (unsigned int)((x - x_min) / span * (float)((1u << bits) - 1));
}

static float uint_to_float(int x_int, float x_min, float x_max, int bits)
{
    float span = x_max - x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + x_min;
}

static void send_can_frame(void)
{
    uint32_t timeout = HAL_GetTick() + 5;
    while (HAL_CAN_GetTxMailboxesFreeLevel(&hcan1) == 0) {
        if (HAL_GetTick() > timeout) return;
    }
    HAL_CAN_AddTxMessage(&hcan1, &txHeader, txData, &txMailbox);
}

// AK60: extended CAN ID, KP/KD/Pos/Vel/Torque byte order
static void send_ak60_cmd(MotorState *m)
{
    uint16_t kp_int = float_to_uint(m->kp,  KP_MIN, KP_MAX, 12);
    uint16_t kd_int = float_to_uint(m->kd,  KD_MIN, KD_MAX, 12);
    uint16_t p_int  = float_to_uint(m->pos + m->pos_offset, AK60_P_MIN, AK60_P_MAX, 16);
    uint16_t v_int  = float_to_uint(m->vel, AK60_V_MIN, AK60_V_MAX, 12);
    uint16_t t_int  = float_to_uint(m->tff, AK60_T_MIN, AK60_T_MAX, 12);

    txData[0] = (kp_int >> 4) & 0xFF;
    txData[1] = ((kp_int & 0xF) << 4) | ((kd_int >> 8) & 0xF);
    txData[2] = kd_int & 0xFF;
    txData[3] = (p_int >> 8) & 0xFF;
    txData[4] = p_int & 0xFF;
    txData[5] = (v_int >> 4) & 0xFF;
    txData[6] = ((v_int & 0xF) << 4) | ((t_int >> 8) & 0xF);
    txData[7] = t_int & 0xFF;

    txHeader.StdId              = 0;
    txHeader.ExtId              = (uint32_t)m->can_id | ((uint32_t)CAN_PACKET_MIT << 8);
    txHeader.IDE                = CAN_ID_EXT;
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 8;
    txHeader.TransmitGlobalTime = DISABLE;
    send_can_frame();
}

// AK70/80: standard CAN ID, Pos/Vel/KP/KD/Torque byte order
static void send_ak70_cmd(MotorState *m)
{
    unsigned int p_int  = float_to_uint(m->pos, AK70_P_MIN, AK70_P_MAX, 16);
    unsigned int v_int  = float_to_uint(m->vel, AK70_V_MIN, AK70_V_MAX, 12);
    unsigned int kp_int = float_to_uint(m->kp,  KP_MIN, KP_MAX, 12);
    unsigned int kd_int = float_to_uint(m->kd,  KD_MIN, KD_MAX, 12);
    unsigned int t_int  = float_to_uint(m->tff, AK70_T_MIN, AK70_T_MAX, 12);

    txData[0] = p_int >> 8;
    txData[1] = p_int & 0xFF;
    txData[2] = v_int >> 4;
    txData[3] = ((v_int & 0xF) << 4) | (kp_int >> 8);
    txData[4] = kp_int & 0xFF;
    txData[5] = kd_int >> 4;
    txData[6] = ((kd_int & 0xF) << 4) | (t_int >> 8);
    txData[7] = t_int & 0xFF;

    txHeader.StdId              = m->can_id;
    txHeader.ExtId              = 0;
    txHeader.IDE                = CAN_ID_STD;
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 8;
    txHeader.TransmitGlobalTime = DISABLE;
    send_can_frame();
}

// AK40: standard CAN ID, same byte order as AK70 but different limits
static void send_ak40_cmd(MotorState *m)
{
    unsigned int p_int  = float_to_uint(m->pos, AK40_P_MIN, AK40_P_MAX, 16);
    unsigned int v_int  = float_to_uint(m->vel, AK40_V_MIN, AK40_V_MAX, 12);
    unsigned int kp_int = float_to_uint(m->kp,  KP_MIN, KP_MAX, 12);
    unsigned int kd_int = float_to_uint(m->kd,  KD_MIN, KD_MAX, 12);
    unsigned int t_int  = float_to_uint(m->tff, AK40_T_MIN, AK40_T_MAX, 12);

    txData[0] = p_int >> 8;
    txData[1] = p_int & 0xFF;
    txData[2] = v_int >> 4;
    txData[3] = ((v_int & 0xF) << 4) | (kp_int >> 8);
    txData[4] = kp_int & 0xFF;
    txData[5] = kd_int >> 4;
    txData[6] = ((kd_int & 0xF) << 4) | (t_int >> 8);
    txData[7] = t_int & 0xFF;

    txHeader.StdId              = m->can_id;
    txHeader.ExtId              = 0;
    txHeader.IDE                = CAN_ID_STD;
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 8;
    txHeader.TransmitGlobalTime = DISABLE;
    send_can_frame();
}

#if TX_ECHO
static void log_tx_cmd(uint8_t motor_idx, MotorState *m)
{
    char buf[80];
    int len = snprintf(buf, sizeof(buf),
        "[TX M%d] pos=%6.3f vel=%5.2f kp=%5.1f kd=%4.2f tff=%5.2f\r\n",
        motor_idx + 1, (double)m->pos, (double)m->vel,
        (double)m->kp, (double)m->kd, (double)m->tff);
    fb_write_main(buf, len);
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

// AK60 set origin: servo mode, extended frame, CAN_PACKET_SET_ORIGIN_HERE (id=5)
// Data[0]=0 → temporary (cleared on power loss); Data[0]=1 → permanent
static void send_ak60_set_origin(MotorState *m)
{
    txData[0] = 0x02;

    txHeader.StdId              = 0;
    txHeader.ExtId              = (uint32_t)m->can_id | ((uint32_t)5 << 8);
    txHeader.IDE                = CAN_ID_EXT;
    txHeader.RTR                = CAN_RTR_DATA;
    txHeader.DLC                = 1;
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

// AK60 feedback: servo-mode format (writes to ring buffer, not UART directly)
static void unpack_ak60_reply(uint8_t motor_idx)
{
    int16_t pos_int = (int16_t)((rxData[0] << 8) | rxData[1]);
    int16_t spd_int = (int16_t)((rxData[2] << 8) | rxData[3]);
    int16_t cur_int = (int16_t)((rxData[4] << 8) | rxData[5]);

    float pos = (float)pos_int * 0.1f;
    float spd = (float)spd_int * 10.0f;
    float cur = (float)cur_int * 0.01f;
    int8_t temp = (int8_t)rxData[6];
    int8_t err  = (int8_t)rxData[7];
    uint32_t ts = HAL_GetTick();

    MotorState *m = uart_cmd_motor(motor_idx);
    m->fb_pos = pos * 0.01745329f - m->pos_offset;
    m->fb_received = 1;
    m->fb_tick = ts;

    char buf[128];
    int len = snprintf(buf, sizeof(buf),
        "[M%d] pos=%.1f deg  spd=%.0f eRPM  I=%.2f A  T=%d  err=%d  t=%lums\r\n",
        motor_idx + 1, (double)(m->fb_pos / 0.01745329f), (double)spd, (double)cur, temp, err, (unsigned long)ts);
    fb_write(buf, len);
}

// AK70/80 feedback: MIT mode format (writes to ring buffer)
static void unpack_ak70_reply(uint8_t motor_idx)
{
    int p_int = (rxData[1] << 8) | rxData[2];
    int v_int = (rxData[3] << 4) | (rxData[4] >> 4);
    int i_int = ((rxData[4] & 0xF) << 8) | rxData[5];

    float pos = uint_to_float(p_int, AK70_P_MIN, AK70_P_MAX, 16);
    float vel = uint_to_float(v_int, AK70_V_MIN, AK70_V_MAX, 12);
    float tau = uint_to_float(i_int, -AK70_T_MAX, AK70_T_MAX, 12);
    int8_t temp = (int8_t)(rxData[6] - 40);
    int8_t err  = (int8_t)rxData[7];
    uint32_t ts = HAL_GetTick();

    MotorState *m = uart_cmd_motor(motor_idx);
    m->fb_pos = pos;
    m->fb_received = 1;
    m->fb_tick = ts;

    char buf[128];
    int len = snprintf(buf, sizeof(buf),
        "[M%d] pos=%.3f rad  vel=%.2f  tau=%.2f  T=%d  err=%d  t=%lums\r\n",
        motor_idx + 1, (double)pos, (double)vel, (double)tau, temp, err, (unsigned long)ts);
    fb_write(buf, len);
}

// AK40 feedback: same format as AK70 but different limits
static void unpack_ak40_reply(uint8_t motor_idx)
{
    int p_int = (rxData[1] << 8) | rxData[2];
    int v_int = (rxData[3] << 4) | (rxData[4] >> 4);
    int i_int = ((rxData[4] & 0xF) << 8) | rxData[5];

    float pos = uint_to_float(p_int, AK40_P_MIN, AK40_P_MAX, 16);
    float vel = uint_to_float(v_int, AK40_V_MIN, AK40_V_MAX, 12);
    float tau = uint_to_float(i_int, -AK40_T_MAX, AK40_T_MAX, 12);
    int8_t temp = (int8_t)(rxData[6] - 40);
    int8_t err  = (int8_t)rxData[7];
    uint32_t ts = HAL_GetTick();

    MotorState *m = uart_cmd_motor(motor_idx);
    m->fb_pos = pos;
    m->fb_received = 1;
    m->fb_tick = ts;

    char buf[128];
    int len = snprintf(buf, sizeof(buf),
        "[M%d] pos=%.3f rad  vel=%.2f  tau=%.2f  T=%d  err=%d  t=%lums\r\n",
        motor_idx + 1, (double)pos, (double)vel, (double)tau, temp, err, (unsigned long)ts);
    fb_write(buf, len);
}

// Identify which motor sent feedback based on CAN ID
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

    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        MotorState *m = uart_cmd_motor(i);
        if (m->can_id == rx_id) {
            if (m->type == MOTOR_TYPE_AK60)
                unpack_ak60_reply(i);
            else if (m->type == MOTOR_TYPE_AK40)
                unpack_ak40_reply(i);
            else
                unpack_ak70_reply(i);
            HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
            return;
        }
    }

    // No match — buffer debug line
    char dbg[64];
    int len = snprintf(dbg, sizeof(dbg), "[M0] unknown CAN id=%d ide=%lu\r\n",
        (int)rx_id, (unsigned long)rxHeader.IDE);
    fb_write(dbg, len);
    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) {
        // Detect "PING\n" and respond immediately with a timestamped PONG.
        // Responding in the ISR (not the ring buffer) keeps the round-trip
        // time as short as possible for accurate clock-offset estimation.
        // "PING\n" is harmless to uart_cmd_feed: 'P'-'1' is out of motor range.
        static char  ping_buf[5];
        static uint8_t ping_idx = 0;
        if (uart_rx_byte == '\n' || uart_rx_byte == '\r') {
            ping_buf[ping_idx] = '\0';
            if (ping_idx == 4 &&
                ping_buf[0] == 'P' && ping_buf[1] == 'I' &&
                ping_buf[2] == 'N' && ping_buf[3] == 'G') {
                char pong[32];
                int plen = snprintf(pong, sizeof(pong), "PONG t=%luus\r\n",
                                    (unsigned long)get_us());
                HAL_UART_Transmit(&huart2, (uint8_t *)pong, (uint16_t)plen, 10);
            }
            ping_idx = 0;
        } else if (ping_idx < 4) {
            ping_buf[ping_idx++] = (char)uart_rx_byte;
        }

        // On an error interrupt the HAL sets ErrorCode before delivering the
        // byte, so a damaged byte can be flagged before its line is parsed.
        // Mark again afterwards: if this byte ended a line, the damaged or
        // lost byte may belong to the next one.
        // Counted here because the re-arm below clears ErrorCode before the
        // HAL calls HAL_UART_ErrorCallback.
        uint32_t rx_err = huart->ErrorCode;
        count_uart_errors(rx_err);
        if (rx_err) uart_cmd_mark_line_corrupt();
        uart_cmd_feed(uart_rx_byte);
        if (rx_err) uart_cmd_mark_line_corrupt();
        HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);
    }
}

// Reached on every UART receive error. When the byte was also delivered,
// HAL_UART_RxCpltCallback has already counted it, re-armed receive and
// cleared ErrorCode. Otherwise (the HAL ended reception on an overrun with
// no byte delivered) count it here and re-arm, so receive never stays off.
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART2) return;
    uint32_t err = huart->ErrorCode;
    count_uart_errors(err);
    if (err) uart_cmd_mark_line_corrupt();
    // The HAL's own SR-then-DR read normally clears ORE; if it is still set,
    // clear it here. The discarded DR byte is part of a line marked corrupt.
    if (__HAL_UART_GET_FLAG(huart, UART_FLAG_ORE))
        __HAL_UART_CLEAR_OREFLAG(huart);
    // Re-arms after ORE (the HAL stopped receive); returns HAL_BUSY, harmlessly,
    // after FE/NE/PE, where receive was never stopped.
    HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);
}

// Report UART receive errors and rejected command lines to the PC, only when
// something changed and at most every UART_REPORT_MIN_MS.
static void report_uart_errors(void)
{
    static uint32_t last_ore, last_fe, last_ne, last_pe, last_rej;
    static uint32_t last_report_ms;
    uint32_t now = HAL_GetTick();
    if (now - last_report_ms < UART_REPORT_MIN_MS) return;

    uint32_t ore = uart_err_ore, fe = uart_err_fe, ne = uart_err_ne, pe = uart_err_pe;
    uint32_t rej = uart_cmd_rejected_count();
    if (ore == last_ore && fe == last_fe && ne == last_ne && pe == last_pe && rej == last_rej)
        return;
    last_ore = ore; last_fe = fe; last_ne = ne; last_pe = pe; last_rej = rej;
    last_report_ms = now;

    char line[112];
    int len = snprintf(line, sizeof(line),
        "[UART] rx errors ore=%lu fe=%lu ne=%lu pe=%lu rejected=%lu t=%lums\r\n",
        (unsigned long)ore, (unsigned long)fe, (unsigned long)ne,
        (unsigned long)pe, (unsigned long)rej, (unsigned long)now);
    fb_write_main(line, len);

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
        fb_write_main(line, len);
    }
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

static void send_mit_cmd(MotorState *m)
{
    if (m->type == MOTOR_TYPE_AK60)
        send_ak60_cmd(m);
    else if (m->type == MOTOR_TYPE_AK40)
        send_ak40_cmd(m);
    else
        send_ak70_cmd(m);
}

static void clamp_tff(MotorState *m)
{
    float t_max;
    if (m->type == MOTOR_TYPE_AK60)
        t_max = AK60_T_MAX;
    else if (m->type == MOTOR_TYPE_AK40)
        t_max = AK40_T_MAX;
    else
        t_max = AK70_T_MAX;
    if (m->tff > t_max) m->tff = t_max;
    if (m->tff < -t_max) m->tff = -t_max;
}

// One main-loop round of frames for every motor except `skip`, used while
// `skip` is being enabled. The enable sequence blocks for ~1.3 s per motor,
// and an AK70/AK40 only replies when it receives a frame, so without this an
// already-enabled motor went silent and the feedback watchdog disabled it
// (M2 on 2026-10-26, while M3 and M4 were being enabled).
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
        } else if (!m->enabled) {
            float saved_kp = m->kp, saved_kd = m->kd, saved_tff = m->tff;
            m->kp = 0.0f; m->kd = 0.0f; m->tff = 0.0f;
            send_mit_cmd(m);
            m->kp = saved_kp; m->kd = saved_kd; m->tff = saved_tff;
        } else {
            clamp_tff(m);
            send_mit_cmd(m);
        }
    }
}

// HAL_Delay for the enable sequence: keeps the other motors serviced at
// least every 10 ms, the main loop's normal rate.
static void enable_delay(uint8_t enabling, uint32_t ms)
{
    while (ms > 0) {
        uint32_t step = ms < 10 ? ms : 10;
        service_other_motors(enabling);
        HAL_Delay(step);
        ms -= step;
    }
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

    // Enable DWT cycle counter for microsecond timestamps
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT       = 0;
    DWT->CTRL        |= DWT_CTRL_CYCCNTENA_Msk;

    HAL_CAN_Start(&hcan1);
    HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING);
    HAL_NVIC_SetPriority(CAN1_RX0_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(CAN1_RX0_IRQn);

    // Servos already started by MX_TIM2_Init (direct register config)
    update_servos();

    uart_cmd_init();
    HAL_NVIC_SetPriority(USART2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
    HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);

    // Enable B1 button (PC13) as emergency stop
    HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

    HAL_Delay(100);

    // Send zero-torque to all motors on startup to suppress power-on snap.
    // Motors that retained MIT mode from a previous session will receive kp=0,
    // kd=0, tff=0 and stay limp instead of snapping to their stale setpoint.
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
        MotorState *m = uart_cmd_motor(i);
        float saved_kp = m->kp;
        float saved_kd = m->kd;
        m->kp  = 0.0f;
        m->kd  = 0.0f;
        m->tff = 0.0f;
        for (int j = 0; j < 3; j++) {
            if (m->type == MOTOR_TYPE_AK60)
                send_ak60_cmd(m);
            else if (m->type == MOTOR_TYPE_AK40)
                send_ak40_cmd(m);
            else
                send_ak70_cmd(m);
            HAL_Delay(10);
        }
        m->kp = saved_kp;
        m->kd = saved_kd;
    }

    /* USER CODE END 2 */

    /* USER CODE BEGIN WHILE */
    while (1)
    {
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
                    m->pos_offset += m->fb_pos;
                } else {
                    send_ak70_set_origin(m);
                }
                HAL_Delay(10);
                m->pos = 0.0f;
                m->fb_pos = 0.0f;
                m->set_origin_pending = 0;
            }

            // Handle pending enable with soft-start. Every wait goes through
            // enable_delay so the other motors keep receiving frames.
            if (m->enable_pending) {
                // AK70/AK40 need MIT mode entry frame
                if (m->type != MOTOR_TYPE_AK60) {
                    for (int j = 0; j < 10; j++) {
                        send_ak70_enable(m);
                        enable_delay(i, 10);
                    }
                    enable_delay(i, 100);
                }

                m->fb_received = 0;
                float kp_target = m->kp;
                float kd_target = m->kd;

                // Probe with zero gain for all motor types so the ramp
                // starts at the actual position instead of 0.
                m->kp  = 0.0f;
                m->kd  = 0.0f;
                m->tff = 0.0f;
                uint32_t probe_end = HAL_GetTick() + 300;
                while (!m->fb_received && HAL_GetTick() < probe_end) {
                    send_mit_cmd(m);
                    enable_delay(i, 10);
                }
                if (m->fb_received)
                    m->pos = m->fb_pos;
                m->kp = kp_target;
                m->kd = kd_target;

                for (int j = 1; j <= 50; j++) {
                    if (m->fb_received && j <= 5)
                        m->pos = m->fb_pos;

                    m->kp = kp_target * (float)j / 50.0f;
                    m->kd = kd_target * (float)j / 50.0f;

                    send_mit_cmd(m);
                    enable_delay(i, 20);
                }
                m->kp = kp_target;
                m->kd = kd_target;

                m->enable_pending = 0;
                m->enabled = 1;
            }

            // Passive feedback mode: poll state via 0xFC, no torque applied
            if (m->fb_only) {
                if (m->type != MOTOR_TYPE_AK60)
                    send_ak70_enable(m);  // 0xFC — triggers state response, no control
                continue;
            }

            // Non-enabled motors get a continuous zero-torque packet so that
            // if they power on after the STM32 they still snap to nothing.
            if (!m->enabled) {
                float saved_kp = m->kp, saved_kd = m->kd, saved_tff = m->tff;
                m->kp = 0.0f; m->kd = 0.0f; m->tff = 0.0f;
                if (m->type == MOTOR_TYPE_AK60)
                    send_ak60_cmd(m);
                else if (m->type == MOTOR_TYPE_AK40)
                    send_ak40_cmd(m);
                else
                    send_ak70_cmd(m);
                m->kp = saved_kp; m->kd = saved_kd; m->tff = saved_tff;
                continue;
            }

            // Feedback watchdog: if this motor lost power or stopped
            // responding, disable it so it doesn't snap on recovery.
            if (m->fb_received &&
                (HAL_GetTick() - m->fb_tick) > FB_WATCHDOG_MS) {
                m->enabled = 0;
                m->tff = 0.0f;
                char wdg[64];
                int wlen = snprintf(wdg, sizeof(wdg),
                    "[M%d] feedback lost — disabled (re-enable required)\r\n",
                    i + 1);
                fb_write_main(wdg, wlen);
                continue;
            }

            // Clamp torque to motor limits
            clamp_tff(m);
            send_mit_cmd(m);

#if TX_ECHO
            log_tx_cmd(i, m);
#endif

            HAL_Delay(2);
        }

        // Update servo PWM outputs
        update_servos();

        report_uart_errors();

        // Drain feedback buffer over UART (non-ISR context).
        // Transmit contiguous chunks to avoid per-byte function call overhead.
        while (fb_tail != fb_head) {
            uint16_t chunk;
            if (fb_head > fb_tail)
                chunk = fb_head - fb_tail;
            else
                chunk = FB_BUF_SIZE - fb_tail;
            HAL_UART_Transmit(&huart2, (uint8_t *)&fb_buf[fb_tail], chunk, 50);
            fb_tail = (fb_tail + chunk) % FB_BUF_SIZE;
        }

        HAL_Delay(2);
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
    huart2.Init.BaudRate     = 1000000;
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
