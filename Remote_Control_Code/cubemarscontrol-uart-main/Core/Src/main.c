/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Robot arm UART remote control (4 motors + 3 servos)
  *
  * Controls up to 4 CubeMars motors (AK60, AK70/80, or AK40) via CAN bus,
  * and 3 MG995 servos via TIM2 PWM (end effector).
  *
  * Protocol: <motor_num><cmd><value>\n
  *   motor_num = 1-4 (Body: Base, Shoulder, Elbow, Linkage)
  *   motor_num = 5-7 (End Effector: Wrist1, Wrist2, Gripper)
  *   P=position (rad for motors, degrees for servos)
  *   V=velocity, K=Kp, D=Kd, T=torque, E=enable, I=CAN_ID, A=type
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

// Position rate limit: max rad/s the commanded position can change
// At 2 rad/s with ~10ms loop, max step per iteration = 0.02 rad
#define POS_RATE_LIMIT  2.0f
#define LOOP_DT         0.012f

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

// Rate-limited commanded position for each motor (ramps toward m->pos)
static float cmd_pos[NUM_MOTORS] = {0};

// Feedback ring buffer — ISR writes, main loop transmits
#define FB_BUF_SIZE 512
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

// AK60 set origin: mode 0x05, data[0]=0x01 (temporary) per CANOriginMode definition
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

    MotorState *m = uart_cmd_motor(motor_idx);
    m->fb_pos = pos * 0.01745329f - m->pos_offset;
    m->fb_received = 1;

    char buf[96];
    int len = snprintf(buf, sizeof(buf),
        "[M%d] pos=%.1f deg  spd=%.0f eRPM  I=%.2f A  T=%d  err=%d\r\n",
        motor_idx + 1, (double)(m->fb_pos / 0.01745329f), (double)spd, (double)cur, temp, err);
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

    MotorState *m = uart_cmd_motor(motor_idx);
    m->fb_pos = pos;
    m->fb_received = 1;

    char buf[96];
    int len = snprintf(buf, sizeof(buf),
        "[M%d] pos=%.3f rad  vel=%.2f  tau=%.2f  T=%d  err=%d\r\n",
        motor_idx + 1, (double)pos, (double)vel, (double)tau, temp, err);
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

    MotorState *m = uart_cmd_motor(motor_idx);
    m->fb_pos = pos;
    m->fb_received = 1;

    char buf[96];
    int len = snprintf(buf, sizeof(buf),
        "[M%d] pos=%.3f rad  vel=%.2f  tau=%.2f  T=%d  err=%d\r\n",
        motor_idx + 1, (double)pos, (double)vel, (double)tau, temp, err);
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
        uart_cmd_feed(uart_rx_byte);
        HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);
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

    HAL_Delay(100);

    /* USER CODE END 2 */

    /* USER CODE BEGIN WHILE */
    while (1)
    {
        for (uint8_t i = 0; i < NUM_MOTORS; i++)
        {
            MotorState *m = uart_cmd_motor(i);

            // Handle set origin
            if (m->set_origin_pending) {
                if (m->type == MOTOR_TYPE_AK60) {
                    m->pos_offset += m->fb_pos;
                    send_ak60_set_origin(m);
                } else {
                    send_ak70_set_origin(m);
                }
                HAL_Delay(10);
                m->pos = 0.0f;
                cmd_pos[i] = 0.0f;
                m->set_origin_pending = 0;
            }

            // Handle pending enable with soft-start
            if (m->enable_pending) {
                // AK70/AK40 need MIT mode entry frame
                if (m->type != MOTOR_TYPE_AK60) {
                    for (int j = 0; j < 10; j++) {
                        send_ak70_enable(m);
                        HAL_Delay(10);
                    }
                    HAL_Delay(100);
                }

                // Soft-start ramp: capture position from first feedback,
                // then ramp gains at that position (matches raw project behavior)
                m->fb_received = 0;
                float kp_target = m->kp;
                float kd_target = m->kd;

                for (int j = 1; j <= 50; j++) {
                    // Use feedback position once available
                    if (m->fb_received && j <= 5) {
                        m->pos = m->fb_pos;
                        cmd_pos[i] = m->fb_pos;
                    }

                    m->kp = kp_target * (float)j / 50.0f;
                    m->kd = kd_target * (float)j / 50.0f;

                    if (m->type == MOTOR_TYPE_AK60)
                        send_ak60_cmd(m);
                    else if (m->type == MOTOR_TYPE_AK40)
                        send_ak40_cmd(m);
                    else
                        send_ak70_cmd(m);

                    HAL_Delay(20);
                }
                m->kp = kp_target;
                m->kd = kd_target;

                m->enable_pending = 0;
                m->enabled = 1;
            }

            // Only send position commands to enabled motors
            if (!m->enabled) continue;

            // Rate-limit position: ramp cmd_pos toward m->pos
            float target = m->pos;
            float error = target - cmd_pos[i];
            float max_step = POS_RATE_LIMIT * LOOP_DT;
            if (error > max_step)
                cmd_pos[i] += max_step;
            else if (error < -max_step)
                cmd_pos[i] -= max_step;
            else
                cmd_pos[i] = target;

            // Send with rate-limited position
            float saved_pos = m->pos;
            m->pos = cmd_pos[i];

            if (m->type == MOTOR_TYPE_AK60)
                send_ak60_cmd(m);
            else if (m->type == MOTOR_TYPE_AK40)
                send_ak40_cmd(m);
            else
                send_ak70_cmd(m);

            m->pos = saved_pos;
            HAL_Delay(2);
        }

        // Update servo PWM outputs
        update_servos();

        // Drain feedback buffer over UART (non-ISR context)
        while (fb_tail != fb_head) {
            uint8_t c = (uint8_t)fb_buf[fb_tail];
            HAL_UART_Transmit(&huart2, &c, 1, 2);
            fb_tail = (fb_tail + 1) % FB_BUF_SIZE;
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
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

    RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState            = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM            = 16;
    RCC_OscInitStruct.PLL.PLLN            = 336;
    RCC_OscInitStruct.PLL.PLLP            = RCC_PLLP_DIV4;
    RCC_OscInitStruct.PLL.PLLQ            = 2;
    RCC_OscInitStruct.PLL.PLLR            = 2;
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
    huart2.Init.BaudRate     = 115200;
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
