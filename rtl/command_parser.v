`timescale 1ns/1ps
`default_nettype none

// A5 5A, opcode, four little-endian u16 fields, RGB332, CRC-8/SMBUS.
module command_parser #(
    parameter integer CLOCK_HZ = 25175000,
    parameter integer TIMEOUT_MS = 10
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  byte_data,
    input  wire        byte_valid,
    input  wire        byte_error,
    output reg         cmd_valid,
    output reg  [15:0] cmd_x,
    output reg  [15:0] cmd_y,
    output reg  [15:0] cmd_width,
    output reg  [15:0] cmd_height,
    output reg  [7:0]  cmd_color
);
    // Split the arithmetic to avoid overflowing a 32-bit parameter expression.
    localparam integer TIMEOUT_CLOCKS = (CLOCK_HZ / 1000) * TIMEOUT_MS
        + ((CLOCK_HZ % 1000) * TIMEOUT_MS + 999) / 1000;
    localparam integer TIMER_BITS = TIMEOUT_CLOCKS > 1 ? $clog2(TIMEOUT_CLOCKS) : 1;
    localparam [1:0] MAGIC_A5 = 0, MAGIC_5A = 1, PAYLOAD = 2, CHECK_CRC = 3;
    reg [1:0] state;
    reg [TIMER_BITS-1:0] timer;
    reg [3:0] index;
    reg [7:0] crc, opcode, color;
    reg [15:0] x, y, width, height;
    wire [16:0] right_edge = {1'b0, x} + {1'b0, width};
    wire [16:0] bottom_edge = {1'b0, y} + {1'b0, height};
    wire in_bounds = x < 640 && y < 480 && width != 0 && height != 0
        && right_edge <= 17'd640 && bottom_edge <= 17'd480;

    function [7:0] crc_next;
        input [7:0] previous;
        input [7:0] value;
        reg [7:0] c;
        integer bit_number;
        begin
            c = previous ^ value;
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1)
                c = c[7] ? (c << 1) ^ 8'h07 : c << 1;
            crc_next = c;
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state <= MAGIC_A5;
            timer <= 0;
            index <= 0;
            crc <= 0;
            opcode <= 0;
            x <= 0; y <= 0; width <= 0; height <= 0; color <= 0;
            cmd_valid <= 0;
            cmd_x <= 0; cmd_y <= 0; cmd_width <= 0; cmd_height <= 0; cmd_color <= 0;
        end else begin
            cmd_valid <= 0;
            if (byte_error) begin
                state <= MAGIC_A5;
                timer <= 0;
            end else if (byte_valid) begin
                timer <= 0;
                if (state == MAGIC_A5 || timer == TIMEOUT_CLOCKS - 1) begin
                    state <= byte_data == 8'ha5 ? MAGIC_5A : MAGIC_A5;
                end else case (state)
                    MAGIC_5A: if (byte_data == 8'h5a) begin
                        index <= 0;
                        crc <= 0;
                        state <= PAYLOAD;
                    end else state <= byte_data == 8'ha5 ? MAGIC_5A : MAGIC_A5;
                    PAYLOAD: begin
                        crc <= crc_next(crc, byte_data);
                        case (index)
                            0: opcode <= byte_data;
                            1: x[7:0] <= byte_data;
                            2: x[15:8] <= byte_data;
                            3: y[7:0] <= byte_data;
                            4: y[15:8] <= byte_data;
                            5: width[7:0] <= byte_data;
                            6: width[15:8] <= byte_data;
                            7: height[7:0] <= byte_data;
                            8: height[15:8] <= byte_data;
                            9: color <= byte_data;
                            default: begin end
                        endcase
                        if (index == 9) state <= CHECK_CRC;
                        else index <= index + 1'b1;
                    end
                    CHECK_CRC: begin
                        if (byte_data == crc && opcode == 8'h01 && in_bounds) begin
                            cmd_x <= x;
                            cmd_y <= y;
                            cmd_width <= width;
                            cmd_height <= height;
                            cmd_color <= color;
                            cmd_valid <= 1;
                        end
                        state <= MAGIC_A5;
                    end
                    default: state <= MAGIC_A5;
                endcase
            end else if (state != MAGIC_A5) begin
                if (timer == TIMEOUT_CLOCKS - 1) begin
                    timer <= 0;
                    state <= MAGIC_A5;
                end else timer <= timer + 1'b1;
            end else timer <= 0;
        end
    end
endmodule
`default_nettype wire
