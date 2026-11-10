`timescale 1ns/1ps
module tb_npu_array;
    localparam integer LANES = 4;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, clear = 0, step = 0;
    reg [LANES*8-1:0] a_data = 0, b_data = 0;
    reg [LANES-1:0] a_valid = 0, b_valid = 0;
    wire [LANES*LANES*32-1:0] sums;
    reg [LANES*LANES*32-1:0] held;
    integer r, c, k, n, length, expected;
    npu_systolic_array #(.LANES(LANES)) dut (.*);

    initial begin
        repeat (3) @(negedge clk); rst = 0;
        for (n = 0; n < 3; n = n + 1) begin
            length = n == 0 ? 1 : (n == 1 ? 3 : 9);
            @(negedge clk); clear = 1; step = 0;
            @(negedge clk); clear = 0; step = 1;
            for (k = 0; k < length; k = k + 1) begin
                a_valid = {LANES{1'b1}}; b_valid = {LANES{1'b1}};
                for (r = 0; r < LANES; r = r + 1) a_data[r*8 +: 8] = r*3 + k - 7;
                for (c = 0; c < LANES; c = c + 1) b_data[c*8 +: 8] = k - c*2 - 4;
                @(negedge clk);
                if (k == 1) begin
                    step = 0; held = sums;
                    repeat (3) @(negedge clk);
                    if (sums !== held) $fatal(1, "array changed during a stall");
                    step = 1;
                end
            end
            a_valid = 0; b_valid = 0;
            repeat (2*LANES) @(negedge clk);
            step = 0;
            for (r = 0; r < LANES; r = r + 1)
                for (c = 0; c < LANES; c = c + 1) begin
                    expected = 0;
                    for (k = 0; k < length; k = k + 1)
                        expected = expected + (r*3 + k - 7)*(k - c*2 - 4);
                    if ($signed(sums[(r*LANES+c)*32 +: 32]) !== expected)
                        $fatal(1, "array K=%0d cell(%0d,%0d) expected %0d got %0d", length, r, c,
                            expected, $signed(sums[(r*LANES+c)*32 +: 32]));
                end
        end
        $display("PASS tb_npu_array: systolic skew, propagation, stalls, K=1/3/9");
        $finish;
    end
    initial begin #100000; $fatal(1, "array timeout"); end
endmodule
