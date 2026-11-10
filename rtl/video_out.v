`timescale 1ns/1ps
`default_nettype none

// Register the complete video bundle together: exactly one pixel of latency.
module video_out (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] rgb_in,
    input  wire       active_in,
    input  wire       hsync_in,
    input  wire       vsync_in,
    output reg  [7:0] rgb,
    output reg        de,
    output reg        hsync,
    output reg        vsync
);
    always @(posedge clk) begin
        if (rst) begin
            rgb <= 0;
            de <= 0;
            hsync <= 1;
            vsync <= 1;
        end else begin
            rgb <= active_in ? rgb_in : 8'h00;
            de <= active_in;
            hsync <= hsync_in;
            vsync <= vsync_in;
        end
    end
endmodule
`default_nettype wire
