`timescale 1ns/1ps
`default_nettype none

module pixel_renderer (
    input  wire [9:0]  x,
    input  wire [9:0]  y,
    input  wire        active,
    input  wire        rect_enable,
    input  wire [15:0] rect_x,
    input  wire [15:0] rect_y,
    input  wire [15:0] rect_width,
    input  wire [15:0] rect_height,
    input  wire [7:0]  rect_color,
    output wire [7:0]  rgb
);
    wire [16:0] right_edge = {1'b0, rect_x} + {1'b0, rect_width};
    wire [16:0] bottom_edge = {1'b0, rect_y} + {1'b0, rect_height};
    wire covered = {7'd0, x} >= {1'b0, rect_x} && {7'd0, x} < right_edge
        && {7'd0, y} >= {1'b0, rect_y} && {7'd0, y} < bottom_edge;
    assign rgb = active && rect_enable && covered ? rect_color : 8'h00;
endmodule
`default_nettype wire
