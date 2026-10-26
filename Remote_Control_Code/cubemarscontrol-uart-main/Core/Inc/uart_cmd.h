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
    float kp;
    float kd;
    float tff;
    uint8_t can_id;
    MotorType type;
    uint8_t enable_pending;
    uint8_t enabled;
    uint8_t set_origin_pending;
    volatile float fb_pos;
    volatile uint8_t fb_received;
    float pos_offset;
} MotorState;

void         uart_cmd_init(void);
void         uart_cmd_feed(uint8_t byte);
MotorState  *uart_cmd_motor(uint8_t idx);
float       *uart_cmd_servo_pos(void);

#endif
