`timescale 1ns/1ps
module tb_npu_linear #(
    parameter integer LANES = 4
);
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1;
    reg wr_valid = 0, wr_sel = 0;
    wire wr_ready;
    reg [15:0] wr_addr = 0;
    reg signed [7:0] wr_data = 0;
    reg scale_wr_valid = 0;
    wire scale_wr_ready;
    reg [15:0] scale_wr_channel = 0;
    reg signed [31:0] scale_wr_bias = 0, scale_wr_multiplier = 1;
    reg [5:0] scale_wr_shift = 0;
    reg cmd_valid = 0, cmd_relu = 0;
    wire cmd_ready;
    reg [15:0] cmd_m = 0, cmd_n = 0, cmd_k = 0;
    wire busy, done, error, out_valid, out_last;
    reg out_ready = 0;
    wire [15:0] out_row, out_col;
    wire signed [7:0] out_data;
    integer expected [0:255];
    reg seen [0:255];
    reg checking = 0, stalled = 0;
    reg random_ready = 0;
    reg [31:0] rng = 32'hb1a52395;
    reg [40:0] held;
    integer outputs = 0, lasts = 0, m = 0, n = 0, r, c, index, last_hold_cycles = 0;
    integer fd, count, case_index, depth, use_relu, k, value;

    npu_linear_top #(.LANES(LANES)) dut (.*);

    always @(negedge clk) begin
        rng = {rng[30:0], rng[31] ^ rng[21] ^ rng[1] ^ rng[0]};
        if (random_ready) begin
            if (out_valid && out_last && last_hold_cycles < 7) begin
                out_ready = 0;
                last_hold_cycles = last_hold_cycles + 1;
            end else out_ready = rng[0] | rng[3];
        end
    end

    always @(posedge clk) begin
        if (rst) stalled = 0;
        else begin
            if (stalled && (!out_valid || {out_last, out_row, out_col, out_data} !== held))
                $fatal(1, "linear output changed under backpressure");
            stalled = out_valid && !out_ready;
            held = {out_last, out_row, out_col, out_data};
            if (checking && out_valid && out_ready) begin
                index = out_row*16 + out_col;
                if (out_row >= m || out_col >= n) $fatal(1, "linear output outside result matrix");
                if (seen[index]) $fatal(1, "duplicate linear result coordinate");
                seen[index] = 1;
                if ($signed(out_data) !== expected[index])
                    $fatal(1, "linear [%0d,%0d]: expected %0d got %0d",
                           out_row, out_col, expected[index], $signed(out_data));
                outputs = outputs + 1;
                if (out_last !== (out_row == m-1 && out_col == n-1))
                    $fatal(1, "linear out_last mismatch");
                if (out_last) lasts = lasts + 1;
            end
        end
    end

    task write_byte(input bit select_b, input [15:0] address, input integer data);
        begin
            @(negedge clk); wr_valid = 1; wr_sel = select_b; wr_addr = address; wr_data = data;
            while (!wr_ready) @(negedge clk);
            @(negedge clk); wr_valid = 0;
        end
    endtask

    task write_scale(input [15:0] channel, input integer bias, multiplier, input [5:0] shift);
        begin
            @(negedge clk); scale_wr_channel = channel; scale_wr_bias = bias;
            scale_wr_multiplier = multiplier; scale_wr_shift = shift; scale_wr_valid = 1;
            while (!scale_wr_ready) @(negedge clk);
            @(posedge clk); #1;
            @(negedge clk); scale_wr_valid = 0;
        end
    endtask

    task reject_command(input integer rows, cols, inner);
        begin
            @(negedge clk); cmd_m = rows; cmd_n = cols; cmd_k = inner; cmd_relu = 0; cmd_valid = 1;
            while (!cmd_ready) @(negedge clk);
            @(posedge clk); #1;
            if (!error || busy) $fatal(1, "invalid/unconfigured descriptor was accepted");
            @(negedge clk); cmd_valid = 0;
        end
    endtask

    task reset_abort(input bit wait_for_output);
        begin
            @(negedge clk); cmd_m = 1; cmd_n = 1; cmd_k = 1; cmd_valid = 1;
            while (!cmd_ready) @(negedge clk);
            @(negedge clk); cmd_valid = 0;
            if (wait_for_output) begin
                wait(out_valid);
                repeat (5) @(negedge clk);
            end
            rst = 1;
            @(negedge clk); rst = 0;
            #1;
            if (busy || out_valid || done || error || !cmd_ready)
                $fatal(1, "reset did not abort linear job/output");
        end
    endtask

    task reject_bad_scale(input [15:0] channel, input integer multiplier, input [5:0] shift);
        begin
            @(negedge clk); scale_wr_channel = channel; scale_wr_bias = 0;
            scale_wr_multiplier = multiplier; scale_wr_shift = shift; scale_wr_valid = 1;
            while (!scale_wr_ready) @(negedge clk);
            @(posedge clk); #1;
            if (!error) $fatal(1, "invalid scale parameter was not reported");
            @(negedge clk); scale_wr_valid = 0;
            @(posedge clk); #1;
            if (error) $fatal(1, "scale error must pulse");
        end
    endtask

    task read_integer(output integer result);
        if ($fscanf(fd, "%d", result) != 1) $fatal(1, "invalid linear vector file");
    endtask

    task run_case(input integer rows, cols, inner, input bit use_relu, hold_first);
        integer item;
        begin
            m = rows; n = cols; outputs = 0; lasts = 0; last_hold_cycles = 0;
            for (item = 0; item < 256; item = item + 1) seen[item] = 0;
            checking = 1; cmd_m = rows; cmd_n = cols; cmd_k = inner; cmd_relu = use_relu;
            @(negedge clk); cmd_valid = 1;
            wr_valid = 1; scale_wr_valid = 1;
            #1;
            if (wr_ready || scale_wr_ready) $fatal(1, "descriptor must have write priority");
            while (!cmd_ready) @(negedge clk);
            @(negedge clk); cmd_valid = 0;
            // A presented write, scale or descriptor cannot alter an active job.
            wr_valid = 1; wr_sel = 0; wr_addr = 0; wr_data = 127;
            scale_wr_valid = 1; scale_wr_channel = 0; scale_wr_bias = 123456;
            cmd_valid = 1; cmd_relu = !use_relu;
            #1;
            if (wr_ready || scale_wr_ready || cmd_ready) $fatal(1, "input accepted while busy");
            @(negedge clk); wr_valid = 0; scale_wr_valid = 0; cmd_valid = 0;
            if (hold_first) begin
                wait(out_valid);
                repeat (5) @(negedge clk);
                if (!busy || cmd_ready || wr_ready || scale_wr_ready)
                    $fatal(1, "linear wrapper accepted input before output drain");
            end
            random_ready = 1;
            while (!done) begin
                if (!busy || cmd_ready || wr_ready || scale_wr_ready)
                    $fatal(1, "task reopened its input window before done");
                @(negedge clk);
            end
            @(negedge clk); checking = 0; random_ready = 0; out_ready = 0;
            if (outputs != rows*cols || lasts != 1 || busy || error)
                $fatal(1, "linear case output count=%0d last=%0d busy=%0d error=%0d",
                       outputs, lasts, busy, error);
            for (item = 0; item < rows; item = item + 1)
                for (index = 0; index < cols; index = index + 1)
                    if (!seen[item*16+index]) $fatal(1, "missing linear result");
            @(negedge clk);
            if (done) $fatal(1, "linear done must pulse");
        end
    endtask

    integer channel;
    initial begin
        repeat (4) @(negedge clk); rst = 0;

        for (channel = 0; channel < 3; channel = channel + 1)
            write_scale(channel, 0, 1, 1);
        reject_bad_scale(0, 0, 1);
        reject_bad_scale(0, -1, 1);
        reject_bad_scale(16, 1, 1);
        reject_bad_scale(0, 1, 32);
        reject_command(0, 1, 1); reject_command(1, 17, 1); reject_command(1, 1, 65);
        reject_command(1, 4, 1);
        write_scale(3, 0, 1, 1);
        write_byte(0, 0, 1);
        write_byte(1, 0, -3); write_byte(1, 1, -1);
        write_byte(1, 2, 1); write_byte(1, 3, 3);
        expected[0] = -2; expected[1] = -1; expected[2] = 1; expected[3] = 2;
        run_case(1, 4, 1, 0, 1);

        write_scale(0, 0, 1, 0); write_scale(1, -200, 1, 0);
        write_scale(2, 200, 1, 0); write_scale(3, 0, 1, 0);
        write_byte(1, 0, -100); write_byte(1, 1, -1);
        write_byte(1, 2, 10); write_byte(1, 3, 127);
        expected[0] = -100; expected[1] = -128; expected[2] = 127; expected[3] = 127;
        run_case(1, 4, 1, 0, 0);

        for (channel = 0; channel < 4; channel = channel + 1)
            write_scale(channel, 0, 1, 0);
        write_byte(1, 0, -5); write_byte(1, 1, -1);
        write_byte(1, 2, 1); write_byte(1, 3, 5);
        expected[0] = 0; expected[1] = 0; expected[2] = 1; expected[3] = 5;
        run_case(1, 4, 1, 1, 0);

        for (channel = 0; channel < 3; channel = channel + 1)
            write_scale(channel, 0, 1, 1);
        write_byte(0, 0, 1); write_byte(0, 64, 2);
        write_byte(1, 0, -3); write_byte(1, 1, -1); write_byte(1, 2, 1);
        expected[0] = -2; expected[1] = -1; expected[2] = 1;
        expected[16] = -3; expected[17] = -1; expected[18] = 1;
        run_case(2, 3, 1, 0, 0);

        write_scale(0, -(1 << 31), 1, 0); write_scale(1, (1 << 31)-1, 1, 0);
        write_byte(0, 0, 1); write_byte(1, 0, -128); write_byte(1, 1, 127);
        expected[0] = -128; expected[1] = 127;
        run_case(1, 2, 1, 0, 0);

        write_scale(0, 0, (1 << 31)-1, 31); write_scale(1, 0, (1 << 31)-1, 31);
        write_byte(1, 0, -1); write_byte(1, 1, 1);
        expected[0] = -1; expected[1] = 1;
        run_case(1, 2, 1, 0, 0);

        // Reset invalidates parameter presence as well as flushing in-flight output.
        reset_abort(0);
        reject_command(1, 1, 1);
        write_scale(0, 0, 1, 0);
        reset_abort(1);
        reject_command(1, 1, 1);
        write_scale(0, 0, 1, 0);
        write_byte(0, 0, 3); write_byte(1, 0, -4);
        expected[0] = -12;
        run_case(1, 1, 1, 0, 1);

        fd = $fopen("build/npu_linear_cases.txt", "r");
        if (!fd) $fatal(1, "linear reference vectors missing");
        read_integer(count);
        for (case_index = 0; case_index < count; case_index = case_index + 1) begin
            read_integer(m); read_integer(n); read_integer(depth); read_integer(use_relu);
            for (r = 0; r < m; r = r + 1)
                for (k = 0; k < depth; k = k + 1) begin
                    read_integer(value); write_byte(0, r*64+k, value);
                end
            for (k = 0; k < depth; k = k + 1)
                for (c = 0; c < n; c = c + 1) begin
                    read_integer(value); write_byte(1, k*16+c, value);
                end
            for (c = 0; c < n; c = c + 1) begin
                read_integer(scale_wr_bias); read_integer(scale_wr_multiplier);
                read_integer(value);
                write_scale(c, scale_wr_bias, scale_wr_multiplier, value);
            end
            for (r = 0; r < m; r = r + 1)
                for (c = 0; c < n; c = c + 1) read_integer(expected[r*16+c]);
            run_case(m, n, depth, use_relu, 1);
        end
        $fclose(fd);

        $display("PASS tb_npu_linear LANES=%0d: %0d Python matrices + 7 directed cases, bias/rounding/ReLU/backpressure/reset", LANES, count);
        $finish;
    end
    initial begin #10000000; $fatal(1, "linear timeout"); end
endmodule
