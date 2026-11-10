`timescale 1ns/1ps
module tb_npu_pe;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, clear = 0, step = 0, a_valid = 0, b_valid = 0;
    reg signed [7:0] a_in = 0, b_in = 0;
    wire signed [7:0] a_out, b_out;
    wire a_valid_out, b_valid_out;
    wire signed [31:0] sum;
    npu_pe dut (.*);

    task tick;
        begin @(posedge clk); #1; end
    endtask
    initial begin
        tick; @(negedge clk); rst = 0; step = 1;
        a_valid = 1; b_valid = 1; a_in = -128; b_in = 127;
        tick;
        if (sum !== -32'sd16256 || a_out !== -8'sd128 || b_out !== 8'sd127 ||
            !a_valid_out || !b_valid_out) $fatal(1, "signed MAC or forwarding mismatch");
        @(negedge clk); step = 0; a_in = 99; b_in = 99;
        repeat (3) tick;
        if (sum !== -32'sd16256 || a_out !== -8'sd128) $fatal(1, "step did not hold state");
        @(negedge clk); step = 1; a_valid = 0;
        tick;
        if (sum !== -32'sd16256 || a_valid_out !== 0) $fatal(1, "invalid A accumulated");
        @(negedge clk); a_valid = 1; b_valid = 0;
        tick;
        if (sum !== -32'sd16256) $fatal(1, "invalid B accumulated");
        @(negedge clk); clear = 1;
        tick;
        if (sum !== 0 || a_valid_out || b_valid_out) $fatal(1, "clear did not flush PE");
        @(negedge clk); clear = 0; a_in = -128; b_in = -128; b_valid = 1;
        repeat (64) tick;
        if (sum !== 32'sd1048576) $fatal(1, "INT32 accumulation was truncated");
        @(negedge clk); rst = 1;
        tick;
        if (sum !== 0 || a_valid_out || b_valid_out) $fatal(1, "reset failed");
        $display("PASS tb_npu_pe: signed extremes, INT32 accumulate, enable, valid and clear");
        $finish;
    end
    initial begin #10000; $fatal(1, "PE timeout"); end
endmodule
