`timescale 1ns/1ps
module tb_uart_rx;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, rx = 1;
    wire [7:0] data;
    wire valid, framing_error;
    integer received = 0, errors = 0;
    reg [7:0] observed [0:15];

    uart_rx #(.CLOCK_HZ(25175000), .BAUD(115200)) dut (
        .clk(clk), .rst(rst), .rx(rx), .data(data),
        .valid(valid), .framing_error(framing_error)
    );
    always @(posedge clk) begin
        if (valid) begin
            observed[received] = data;
            received = received + 1;
        end
        if (framing_error) errors = errors + 1;
    end

    // Independent transmit clock: verify nominal and +/- 1.4% baud skew.
    task send_byte(input [7:0] value, input integer bit_cycles, input bit good_stop);
        integer i;
        begin
            @(negedge clk); rx = 0;
            repeat (bit_cycles) @(negedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rx = value[i];
                repeat (bit_cycles) @(negedge clk);
            end
            rx = good_stop;
            repeat (bit_cycles) @(negedge clk);
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        rst = 0;
        // A short low glitch must not turn into a byte.
        rx = 0; repeat (20) @(negedge clk); rx = 1;
        repeat (300) @(negedge clk);
        if (received != 0) $fatal(1, "false start accepted");
        send_byte(8'h00, 219, 1);
        send_byte(8'hff, 219, 1);
        send_byte(8'ha5, 216, 1);
        send_byte(8'h5a, 222, 1);
        repeat (10) @(negedge clk);
        if (received != 4 || observed[0] !== 8'h00 || observed[1] !== 8'hff ||
            observed[2] !== 8'ha5 || observed[3] !== 8'h5a)
            $fatal(1, "UART data/skew mismatch: %0d bytes", received);
        send_byte(8'h3c, 219, 0);
        // A break must not emit data; give the receiver idle time to recover.
        repeat (2600) @(negedge clk);
        rx = 1; repeat (300) @(negedge clk);
        if (received != 4 || errors == 0) $fatal(1, "bad stop accepted or not reported");
        send_byte(8'h96, 219, 1);
        repeat (10) @(negedge clk);
        if (received != 5 || observed[4] !== 8'h96) $fatal(1, "framing recovery failed");
        // Reset halfway through a character, then receive another one.
        rx = 0; repeat (600) @(negedge clk); rst = 1;
        repeat (4) @(negedge clk); rx = 1; rst = 0;
        repeat (300) @(negedge clk);
        send_byte(8'h81, 219, 1);
        repeat (10) @(negedge clk);
        if (received != 6 || observed[5] !== 8'h81) $fatal(1, "reset recovery failed");
        $display("PASS tb_uart_rx: bytes, baud skew, false start, framing and reset");
        $finish;
    end
    initial begin #1000000; $fatal(1, "UART test timeout"); end
endmodule
