`timescale 1ns/1ps
module tb_command_parser;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, byte_valid = 0, byte_error = 0;
    reg [7:0] byte_data = 0;
    wire cmd_valid;
    wire [15:0] cmd_x, cmd_y, cmd_width, cmd_height;
    wire [7:0] cmd_color;
    integer accepted = 0;
    reg [7:0] packet [0:12];

    command_parser #(.CLOCK_HZ(1000), .TIMEOUT_MS(10)) dut (
        .clk(clk), .rst(rst), .byte_data(byte_data),
        .byte_valid(byte_valid), .byte_error(byte_error), .cmd_valid(cmd_valid),
        .cmd_x(cmd_x), .cmd_y(cmd_y), .cmd_width(cmd_width),
        .cmd_height(cmd_height), .cmd_color(cmd_color)
    );
    always @(posedge clk) if (cmd_valid) accepted = accepted + 1;

    function automatic [7:0] crc_step(input [7:0] old, input [7:0] value);
        reg [7:0] c;
        integer bit_index;
        begin
            c = old ^ value;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                c = c[7] ? (c << 1) ^ 8'h07 : c << 1;
            crc_step = c;
        end
    endfunction
    task send_byte(input [7:0] value);
        begin
            @(negedge clk); byte_data = value; byte_valid = 1;
            @(negedge clk); byte_valid = 0;
        end
    endtask
    task make_packet(input [7:0] op, input [15:0] x, y, w, h, input [7:0] color);
        integer i;
        reg [7:0] crc;
        begin
            packet[0] = 8'ha5; packet[1] = 8'h5a; packet[2] = op;
            packet[3] = x[7:0]; packet[4] = x[15:8];
            packet[5] = y[7:0]; packet[6] = y[15:8];
            packet[7] = w[7:0]; packet[8] = w[15:8];
            packet[9] = h[7:0]; packet[10] = h[15:8]; packet[11] = color;
            crc = 0;
            for (i = 2; i < 12; i = i + 1) crc = crc_step(crc, packet[i]);
            packet[12] = crc;
        end
    endtask
    task transmit;
        integer i;
        begin
            for (i = 0; i < 13; i = i + 1) send_byte(packet[i]);
            repeat (3) @(negedge clk);
        end
    endtask
    task expect_count(input integer n);
        if (accepted != n) $fatal(1, "accepted %0d packets, expected %0d", accepted, n);
    endtask

    initial begin
        repeat (3) @(negedge clk); rst = 0;
        send_byte(8'h12); send_byte(8'ha5); send_byte(8'ha5);
        make_packet(1, 300, 260, 320, 200, 8'h1f); transmit;
        expect_count(1);
        if (cmd_x !== 300 || cmd_y !== 260 || cmd_width !== 320 ||
            cmd_height !== 200 || cmd_color !== 8'h1f) $fatal(1, "decoded fields mismatch");
        make_packet(1, 0, 0, 640, 480, 8'hff); transmit; expect_count(2);
        make_packet(1, 639, 479, 1, 1, 8'ha5); transmit; expect_count(3);
        make_packet(1, 0, 0, 0, 1, 0); transmit; expect_count(3);
        make_packet(1, 0, 0, 1, 0, 0); transmit; expect_count(3);
        make_packet(1, 640, 0, 1, 1, 0); transmit; expect_count(3);
        make_packet(1, 0, 480, 1, 1, 0); transmit; expect_count(3);
        make_packet(1, 639, 0, 2, 1, 0); transmit; expect_count(3);
        make_packet(1, 0, 479, 1, 2, 0); transmit; expect_count(3);
        make_packet(1, 65535, 0, 65535, 1, 0); transmit; expect_count(3);
        make_packet(1, 0, 65535, 1, 65535, 0); transmit; expect_count(3);
        make_packet(2, 1, 2, 3, 4, 0); transmit; expect_count(3);
        make_packet(1, 1, 2, 3, 4, 0); packet[12] = packet[12] ^ 1;
        transmit; expect_count(3);

        // Partial packet must time out, and a later complete packet must work.
        send_byte(8'ha5); send_byte(8'h5a); send_byte(1);
        repeat (12) @(negedge clk);
        make_packet(1, 20, 30, 40, 50, 8'h03); transmit; expect_count(4);
        // A receive framing error discards any partial command.
        send_byte(8'ha5); send_byte(8'h5a);
        @(negedge clk); byte_error = 1;
        @(negedge clk); byte_error = 0;
        make_packet(1, 21, 31, 41, 51, 8'h55); transmit; expect_count(5);
        send_byte(8'ha5); send_byte(8'h5a); send_byte(1);
        @(negedge clk); rst = 1;
        @(negedge clk); rst = 0;
        make_packet(1, 22, 32, 42, 52, 8'haa); transmit; expect_count(6);
        if (cmd_x !== 22 || cmd_y !== 32 || cmd_width !== 42 ||
            cmd_height !== 52 || cmd_color !== 8'haa) $fatal(1, "parser recovery mismatch");
        $display("PASS tb_command_parser: CRC, ranges, overflow, noise, timeout and recovery");
        $finish;
    end
    initial begin #100000; $fatal(1, "parser test timeout"); end
endmodule
