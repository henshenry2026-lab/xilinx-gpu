`timescale 1ns/1ps
`default_nettype none

// 640x480 @ 59.94 Hz with a 25.175 MHz pixel clock, negative HS/VS.
module video_timing (
    input  wire       clk,
    input  wire       rst,
    output wire [9:0] x,
    output wire [9:0] y,
    output wire       active,
    output wire       hsync,
    output wire       vsync,
    output wire       frame_commit
);
    reg [9:0] h_count, v_count;
    always @(posedge clk) begin
        if (rst) begin
            h_count <= 0;
            v_count <= 0;
        end else if (h_count == 799) begin
            h_count <= 0;
            if (v_count == 524) v_count <= 0;
            else v_count <= v_count + 1'b1;
        end else h_count <= h_count + 1'b1;
    end

    assign x = h_count;
    assign y = v_count;
    assign active = h_count < 640 && v_count < 480;
    assign hsync = !(h_count >= 656 && h_count < 752);
    assign vsync = !(v_count >= 490 && v_count < 492);
    assign frame_commit = h_count == 0 && v_count == 480;
endmodule
`default_nettype wire
