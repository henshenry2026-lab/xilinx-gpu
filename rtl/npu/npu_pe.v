`timescale 1ns/1ps
`default_nettype none

// Output-stationary signed INT8 MAC, forwarding A east and B south.
module npu_pe (
    input wire clk, rst, clear, step,
    input wire signed [7:0] a_in, b_in,
    input wire a_valid, b_valid,
    output reg signed [7:0] a_out, b_out,
    output reg a_valid_out, b_valid_out,
    output reg signed [31:0] sum
);
    (* use_dsp = "yes" *) wire signed [15:0] product = a_in * b_in;
    always @(posedge clk) begin
        if (rst || clear) begin
            a_out <= 0; b_out <= 0;
            a_valid_out <= 0; b_valid_out <= 0;
            sum <= 0;
        end else if (step) begin
            a_out <= a_in; b_out <= b_in;
            a_valid_out <= a_valid; b_valid_out <= b_valid;
            if (a_valid && b_valid) sum <= sum + {{16{product[15]}}, product};
        end
    end
endmodule
`default_nettype wire
