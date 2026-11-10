`timescale 1ns/1ps
`default_nettype none

module gpu_top #(
    parameter integer CLOCK_HZ = 25175000,
    parameter integer BAUD = 115200
) (
    input  wire       clk_pix,
    input  wire       rst,
    input  wire       uart_rx_pin,
    output wire [7:0] rgb,
    output wire       hsync,
    output wire       vsync,
    output wire       de
);
    wire [7:0] byte_data, cmd_color, rect_color, pixel_rgb;
    wire byte_valid, byte_error, cmd_valid, rect_enable;
    wire [15:0] cmd_x, cmd_y, cmd_width, cmd_height;
    wire [15:0] rect_x, rect_y, rect_width, rect_height;
    wire [9:0] x, y;
    wire active, raw_hsync, raw_vsync, frame_commit;

    uart_rx #(.CLOCK_HZ(CLOCK_HZ), .BAUD(BAUD)) receiver (
        .clk(clk_pix), .rst(rst), .rx(uart_rx_pin), .data(byte_data),
        .valid(byte_valid), .framing_error(byte_error)
    );
    command_parser #(.CLOCK_HZ(CLOCK_HZ)) parser (
        .clk(clk_pix), .rst(rst), .byte_data(byte_data), .byte_valid(byte_valid),
        .byte_error(byte_error), .cmd_valid(cmd_valid), .cmd_x(cmd_x), .cmd_y(cmd_y),
        .cmd_width(cmd_width), .cmd_height(cmd_height), .cmd_color(cmd_color)
    );
    scene_regs scene (
        .clk(clk_pix), .rst(rst), .frame_commit(frame_commit), .cmd_valid(cmd_valid),
        .cmd_x(cmd_x), .cmd_y(cmd_y), .cmd_width(cmd_width), .cmd_height(cmd_height),
        .cmd_color(cmd_color), .rect_enable(rect_enable), .rect_x(rect_x), .rect_y(rect_y),
        .rect_width(rect_width), .rect_height(rect_height), .rect_color(rect_color)
    );
    video_timing timing (
        .clk(clk_pix), .rst(rst), .x(x), .y(y), .active(active),
        .hsync(raw_hsync), .vsync(raw_vsync), .frame_commit(frame_commit)
    );
    pixel_renderer renderer (
        .x(x), .y(y), .active(active), .rect_enable(rect_enable),
        .rect_x(rect_x), .rect_y(rect_y), .rect_width(rect_width), .rect_height(rect_height),
        .rect_color(rect_color), .rgb(pixel_rgb)
    );
    video_out output_stage (
        .clk(clk_pix), .rst(rst), .rgb_in(pixel_rgb), .active_in(active),
        .hsync_in(raw_hsync), .vsync_in(raw_vsync), .rgb(rgb), .de(de),
        .hsync(hsync), .vsync(vsync)
    );
endmodule
`default_nettype wire
