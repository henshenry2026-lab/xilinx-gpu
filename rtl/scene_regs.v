`timescale 1ns/1ps
`default_nettype none

module scene_regs (
    input  wire        clk,
    input  wire        rst,
    input  wire        frame_commit,
    input  wire        cmd_valid,
    input  wire [15:0] cmd_x,
    input  wire [15:0] cmd_y,
    input  wire [15:0] cmd_width,
    input  wire [15:0] cmd_height,
    input  wire [7:0]  cmd_color,
    output reg         rect_enable,
    output reg  [15:0] rect_x,
    output reg  [15:0] rect_y,
    output reg  [15:0] rect_width,
    output reg  [15:0] rect_height,
    output reg  [7:0]  rect_color
);
    reg pending_valid;
    reg [15:0] pending_x, pending_y, pending_width, pending_height;
    reg [7:0] pending_color;

    always @(posedge clk) begin
        if (rst) begin
            pending_valid <= 0;
            pending_x <= 0; pending_y <= 0;
            pending_width <= 0; pending_height <= 0; pending_color <= 0;
            rect_enable <= 0;
            rect_x <= 0; rect_y <= 0; rect_width <= 0; rect_height <= 0; rect_color <= 0;
        end else begin
            if (frame_commit && pending_valid) begin
                rect_enable <= 1;
                rect_x <= pending_x;
                rect_y <= pending_y;
                rect_width <= pending_width;
                rect_height <= pending_height;
                rect_color <= pending_color;
                pending_valid <= 0;
            end
            // A simultaneous command remains pending for the following commit.
            if (cmd_valid) begin
                pending_x <= cmd_x;
                pending_y <= cmd_y;
                pending_width <= cmd_width;
                pending_height <= cmd_height;
                pending_color <= cmd_color;
                pending_valid <= 1;
            end
        end
    end
endmodule
`default_nettype wire
