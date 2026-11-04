#ifndef UART_CMD_H
#define UART_CMD_H

#include <stdint.h>

#define NUM_MOTORS 4
// End effector: one wrist servo and one pinch (gripper) servo.  Their
// command digits stay '5' and '7' from the original three-servo layout
// (Wrist 1, Wrist 2, Gripper); '6' (old Wrist 2 address) is ignored.
#define NUM_SERVOS     2
#define SERVO_WRIST    0   // command '5', PA0 / TIM2_CH1
#define SERVO_GRIPPER  1   // command '7', PA1 / TIM2_CH2
#define SERVO_CMD_SLOTS 3  // command digits '5'..'7' are servo addresses

// Gripper travel, in the firmware's servo unit: degrees from centre at
// 8.333 us/deg, which is (servo angle - 90) * 4/3.  The gripper must not go
// below servo 156 deg; 180 is its full extension.  Enforced here so that every
// sender is limited.  The PC controller has the same limits (SERVO_LIMITS_DEG,
// SERVO_PARK_DEG), so change both together.
#define SERVO_GRIPPER_CMD_MIN  88.1f   // 2234 us = 156.1 deg; 88.0 would truncate to just under 156
#define SERVO_GRIPPER_CMD_MAX  120.0f  // servo 180 deg
#define SERVO_GRIPPER_CMD_PARK 104.0f  // servo 168 deg, mid-travel: boot position

typedef enum {
    MOTOR_TYPE_AK40 = 40,
    MOTOR_TYPE_AK60 = 60,
    MOTOR_TYPE_AK70 = 70,
} MotorType;

typedef struct {
    float pos;
    float vel;
    float kp;           // inner MIT spring gain (set via K command)
    float kd;           // inner MIT damper gain (set via D command)
    float tff;          // torque feedforward (set via T command)
    uint8_t can_id;
    MotorType type;
    uint8_t enable_pending;
    uint8_t enabled;
    uint8_t set_origin_pending;
    volatile float fb_pos;
    volatile uint8_t fb_received;
    volatile uint32_t fb_tick;
    float pos_offset;            // AK60 software origin (rad), added to every P sent
    volatile int16_t fb_raw_count;     // AK60 latest raw position, 0.1 deg counts
    volatile int32_t pos_offset_counts; // AK60 software origin, 0.1 deg counts
    uint8_t fb_only;     // 1 = poll state via 0xFC, no torque (set via R command)
} MotorState;

// One MIT command, copied out of MotorState in one piece (uart_cmd_read).
typedef struct {
    float pos;
    float vel;
    float kp;
    float kd;
    float tff;
} MotorCmd;

void         uart_cmd_init(void);
void         uart_cmd_feed(uint8_t byte);
MotorState  *uart_cmd_motor(uint8_t idx);
void         uart_cmd_read(uint8_t idx, MotorCmd *out);  // atomic w.r.t. the UART ISR
float       *uart_cmd_servo_pos(void);
uint8_t      uart_cmd_e_stop_requested(void);
void         uart_cmd_clear_e_stop(void);

// Line validation. A line is rejected (and its command ignored) when a UART
// error hit one of its bytes, it overflowed the buffer, its checksum is wrong
// or missing (value commands), or its value is not a complete finite number.
// Room for "1M-12.5000,-45.500,500.00,5.0000,-18.0000*HH".
#define UART_CMD_LINE_MAX 64
void         uart_cmd_mark_line_corrupt(void);   // call from the UART error ISR
uint32_t     uart_cmd_rejected_count(void);
uint8_t      uart_cmd_take_rejected(char *out, uint8_t out_size);  // 1 if a new rejected line was copied

#endif
