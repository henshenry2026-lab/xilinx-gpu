`timescale 1ns/1ps
module tb_scene;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, cmd_valid = 0, frame_commit = 0;
    reg [15:0] cmd_x = 0, cmd_y = 0, cmd_width = 0, cmd_height = 0;
    reg [7:0] cmd_color = 0;
    wire rect_enable;
    wire [15:0] rect_x, rect_y, rect_width, rect_height;
    wire [7:0] rect_color, rgb;
    reg [9:0] x = 0, y = 0;
    reg active = 1;

    scene_regs dut (.*);
    pixel_renderer renderer (.*);

    task command(input [15:0] px, py, w, h, input [7:0] color);
        begin
            @(negedge clk);
            cmd_valid = 1; cmd_x = px; cmd_y = py;
            cmd_width = w; cmd_height = h; cmd_color = color;
            @(negedge clk); cmd_valid = 0;
        end
    endtask
    task commit;
        begin
            @(negedge clk); frame_commit = 1;
            @(negedge clk); frame_commit = 0;
        end
    endtask
    task pixel(input [9:0] px, py, input [7:0] expected);
        begin
            x = px; y = py; #1;
            if (rgb !== expected) $fatal(1, "pixel (%0d,%0d): %h expected %h", x, y, rgb, expected);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk); rst = 0;
        pixel(0, 0, 0);
        if (rect_enable !== 0) $fatal(1, "scene enabled after reset");
        command(10, 20, 30, 40, 8'he0);
        pixel(10, 20, 0);
        command(100, 80, 20, 10, 8'h1c);
        commit;
        if (!rect_enable || rect_x !== 100 || rect_y !== 80) $fatal(1, "last command not committed");
        pixel(100, 80, 8'h1c); pixel(119, 89, 8'h1c);
        pixel(99, 80, 0); pixel(120, 80, 0); pixel(100, 79, 0); pixel(100, 90, 0);
        active = 0; pixel(100, 80, 0); active = 1;
        commit; pixel(100, 80, 8'h1c);

        // Collision semantics: old pending scene commits; simultaneous new one waits.
        command(5, 6, 7, 8, 8'h03);
        @(negedge clk);
        frame_commit = 1; cmd_valid = 1;
        cmd_x = 200; cmd_y = 100; cmd_width = 40; cmd_height = 20; cmd_color = 8'hff;
        @(negedge clk); frame_commit = 0; cmd_valid = 0;
        pixel(5, 6, 8'h03); pixel(200, 100, 0);
        commit; pixel(5, 6, 0); pixel(200, 100, 8'hff);
        @(negedge clk); rst = 1;
        @(negedge clk); rst = 0;
        pixel(200, 100, 0);
        if (rect_enable !== 0) $fatal(1, "reset did not clear scene");
        $display("PASS tb_scene: atomic commit, latest command, collision and pixel edges");
        $finish;
    end
    initial begin #10000; $fatal(1, "scene test timeout"); end
endmodule
