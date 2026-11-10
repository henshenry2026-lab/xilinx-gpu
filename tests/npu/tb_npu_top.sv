`timescale 1ns/1ps
module tb_npu_top #(
    parameter integer LANES = 4
);
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1;
    reg wr_valid = 0, wr_sel = 0;
    wire wr_ready;
    reg [15:0] wr_addr = 0;
    reg signed [7:0] wr_data = 0;
    reg cmd_valid = 0;
    wire cmd_ready;
    reg [15:0] cmd_m = 0, cmd_n = 0, cmd_k = 0;
    wire busy, done, error, out_valid, out_last;
    reg out_ready = 0;
    wire [15:0] out_row, out_col;
    wire signed [31:0] out_data;
    reg checking = 0, random_ready = 0, stalled = 0;
    reg [31:0] rng = 32'h2395cafe;
    reg [64:0] held;
    integer fd, count, case_index, m, n, depth, r, c, k, value, outputs, last_count;
    integer av [0:1023], bv [0:1023], golden [0:255];
    reg seen [0:255];
    npu_top #(.LANES(LANES)) dut (.*);

    always @(negedge clk) begin
        rng = {rng[30:0], rng[31] ^ rng[21] ^ rng[1] ^ rng[0]};
        if (random_ready) out_ready = rng[0] | rng[3];
        else out_ready = 0;
    end
    always @(posedge clk) begin
        if (rst) stalled = 0;
        else begin
            if (stalled && {out_last, out_row, out_col, out_data} !== held)
                $fatal(1, "output changed while valid and not ready");
            if (stalled && !out_valid) $fatal(1, "valid dropped under backpressure");
            stalled = out_valid && !out_ready;
            held = {out_last, out_row, out_col, out_data};
            if (checking && out_valid && out_ready) begin
                if (out_row >= m || out_col >= n) $fatal(1, "invalid tail tile output");
                if (seen[out_row*16+out_col]) $fatal(1, "duplicate result coordinate");
                seen[out_row*16+out_col] = 1;
                if (out_data !== golden[out_row*16+out_col])
                    $fatal(1, "case %0d [%0d,%0d]: expected %0d got %0d", case_index,
                        out_row, out_col, golden[out_row*16+out_col], out_data);
                if (out_last !== (out_row == m-1 && out_col == n-1)) $fatal(1, "out_last mismatch");
                outputs = outputs + 1;
                if (out_last) last_count = last_count + 1;
            end
        end
    end
    task read_integer(output integer result);
        if ($fscanf(fd, "%d", result) != 1) $fatal(1, "invalid NPU vector file");
    endtask
    task write_byte(input bit select_b, input [15:0] address, input integer data);
        begin
            @(negedge clk); wr_valid = 1; wr_sel = select_b; wr_addr = address; wr_data = data;
            while (!wr_ready) @(negedge clk);
            @(negedge clk); wr_valid = 0;
            if (address % 7 == 3) repeat (2) @(negedge clk);
        end
    endtask
    task launch(input [15:0] rows, cols, inner);
        begin
            @(negedge clk); cmd_m = rows; cmd_n = cols; cmd_k = inner; cmd_valid = 1;
            while (!cmd_ready) @(negedge clk);
            @(negedge clk); cmd_valid = 0;
        end
    endtask
    task invalid_command(input [15:0] rows, cols, inner);
        begin
            launch(rows, cols, inner);
            if (!error || busy || done || out_valid) $fatal(1, "invalid descriptor accepted");
            @(negedge clk);
            if (error) $fatal(1, "error must pulse for one cycle");
        end
    endtask

    initial begin
        repeat (4) @(negedge clk); rst = 0;
        invalid_command(0, 1, 1); invalid_command(1, 0, 1); invalid_command(1, 1, 0);
        invalid_command(17, 1, 1); invalid_command(1, 17, 1); invalid_command(1, 1, 65);
        write_byte(0, 1024, 55);
        if (!error || busy) $fatal(1, "invalid A write not reported");
        write_byte(1, 1024, 55);
        if (!error || busy) $fatal(1, "invalid B write not reported");

        // Reset aborts both compute and a blocked result stream. Reload before use.
        write_byte(0, 0, 3); write_byte(1, 0, -4);
        launch(1, 1, 1);
        @(negedge clk); rst = 1;
        @(negedge clk); rst = 0;
        if (busy || out_valid || done) $fatal(1, "reset did not abort compute");
        write_byte(0, 0, 3); write_byte(1, 0, -4);
        launch(1, 1, 1);
        wait(out_valid); repeat (5) @(negedge clk);
        rst = 1; @(negedge clk); rst = 0;
        if (busy || out_valid || done) $fatal(1, "reset did not abort blocked output");

        fd = $fopen("build/npu_cases.txt", "r");
        if (!fd) $fatal(1, "NPU vector file missing");
        read_integer(count);
        for (case_index = 0; case_index < count; case_index = case_index + 1) begin
            read_integer(m); read_integer(n); read_integer(depth);
            for (r = 0; r < m; r = r + 1)
                for (k = 0; k < depth; k = k + 1) read_integer(av[r*64+k]);
            for (k = 0; k < depth; k = k + 1)
                for (c = 0; c < n; c = c + 1) read_integer(bv[k*16+c]);
            for (r = 0; r < m; r = r + 1)
                for (c = 0; c < n; c = c + 1) read_integer(golden[r*16+c]);
            for (r = 0; r < 256; r = r + 1) seen[r] = 0;
            outputs = 0; last_count = 0;
            for (r = 0; r < m; r = r + 1)
                for (k = 0; k < depth; k = k + 1) write_byte(0, r*64+k, av[r*64+k]);
            for (k = 0; k < depth; k = k + 1)
                for (c = 0; c < n; c = c + 1) write_byte(1, k*16+c, bv[k*16+c]);
            checking = 1;
            launch(m, n, depth);
            // While busy, neither a memory write nor another descriptor can handshake.
            @(negedge clk); wr_valid = 1; wr_sel = 1; wr_addr = 0; wr_data = 99; cmd_valid = 1;
            if (wr_ready || cmd_ready) $fatal(1, "input accepted while busy");
            @(negedge clk); wr_valid = 0; cmd_valid = 0;
            wait(out_valid);
            repeat (8) @(negedge clk);
            random_ready = 1;
            wait(done); @(negedge clk);
            checking = 0; random_ready = 0;
            if (outputs != m*n || last_count != 1 || busy || error)
                $fatal(1, "case %0d: count %0d expected %0d, last=%0d", case_index, outputs, m*n, last_count);
            for (r = 0; r < m; r = r + 1)
                for (c = 0; c < n; c = c + 1)
                    if (!seen[r*16+c]) $fatal(1, "missing result");
            @(negedge clk);
            if (done) $fatal(1, "done must pulse for one cycle");
        end
        $fclose(fd);
        $display("PASS tb_npu_top LANES=%0d: %0d matrices, signed/tails/backpressure/reset/errors", LANES, count);
        $finish;
    end
    initial begin #10000000; $fatal(1, "NPU integration timeout"); end
endmodule
