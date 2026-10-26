#ifndef UART_CMD_H
#define UART_CMD_H

#include <stdint.h>

#define NUM_MOTORS 4
#define NUM_SERVOS 3

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
    float pos_offset;
    uint8_t fb_only;     // 1 = poll state via 0xFC, no torque (set via R command)
} MotorState;

void         uart_cmd_init(void);
void         uart_cmd_feed(uint8_t byte);
MotorState  *uart_cmd_motor(uint8_t idx);
float       *uart_cmd_servo_pos(void);
uint8_t      uart_cmd_e_stop_requested(void);
void         uart_cmd_clear_e_stop(void);

// Line validation. A line is rejected (and its command ignored) when a UART
// error hit one of its bytes, it overflowed the buffer, or its value is not a
// complete finite number.
#define UART_CMD_LINE_MAX 32
void         uart_cmd_mark_line_corrupt(void);   // call from the UART error ISR
uint32_t     uart_cmd_rejected_count(void);
uint8_t      uart_cmd_take_rejected(char *out, uint8_t out_size);  // 1 if a new rejected line was copied

#endif
