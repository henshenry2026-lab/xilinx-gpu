`timescale 1ns/1ps
`default_nettype none

// One vector pair per step. Lane r/c is skewed by r/c cycles before the grid.
module npu_systolic_array #(
    parameter integer LANES = 4
) (
    input wire clk, rst, clear, step,
    input wire [LANES*8-1:0] a_data, b_data,
    input wire [LANES-1:0] a_valid, b_valid,
    output wire [LANES*LANES*32-1:0] sums
);
    wire signed [7:0] a_link [0:LANES*(LANES+1)-1];
    wire signed [7:0] b_link [0:LANES*(LANES+1)-1];
    wire av_link [0:LANES*(LANES+1)-1];
    wire bv_link [0:LANES*(LANES+1)-1];
    genvar lane, row, col;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin: skew
            if (lane == 0) begin: direct
                assign a_link[0] = a_data[7:0];
                assign b_link[0] = b_data[7:0];
                assign av_link[0] = a_valid[0];
                assign bv_link[0] = b_valid[0];
            end else begin: delayed
                reg [lane*8-1:0] a_pipe, b_pipe;
                reg [lane-1:0] av_pipe, bv_pipe;
                always @(posedge clk) begin
                    if (rst || clear) begin
                        a_pipe <= 0; b_pipe <= 0; av_pipe <= 0; bv_pipe <= 0;
                    end else if (step) begin
                        a_pipe <= (a_pipe << 8) | a_data[lane*8 +: 8];
                        b_pipe <= (b_pipe << 8) | b_data[lane*8 +: 8];
                        av_pipe <= (av_pipe << 1) | a_valid[lane];
                        bv_pipe <= (bv_pipe << 1) | b_valid[lane];
                    end
                end
                assign a_link[lane*(LANES+1)] = a_pipe[(lane-1)*8 +: 8];
                assign b_link[lane] = b_pipe[(lane-1)*8 +: 8];
                assign av_link[lane*(LANES+1)] = av_pipe[lane-1];
                assign bv_link[lane] = bv_pipe[lane-1];
            end
        end
        for (row = 0; row < LANES; row = row + 1) begin: rows
            for (col = 0; col < LANES; col = col + 1) begin: cols
                npu_pe pe (
                    .clk(clk), .rst(rst), .clear(clear), .step(step),
                    .a_in(a_link[row*(LANES+1)+col]), .b_in(b_link[row*LANES+col]),
                    .a_valid(av_link[row*(LANES+1)+col]), .b_valid(bv_link[row*LANES+col]),
                    .a_out(a_link[row*(LANES+1)+col+1]), .b_out(b_link[(row+1)*LANES+col]),
                    .a_valid_out(av_link[row*(LANES+1)+col+1]), .b_valid_out(bv_link[(row+1)*LANES+col]),
                    .sum(sums[(row*LANES+col)*32 +: 32])
                );
            end
        end
    endgenerate
endmodule
`default_nettype wire
