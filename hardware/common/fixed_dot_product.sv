module fixed_dot_product #(
    parameter IN_WIDTH  = 32,
    parameter W_WIDTH   = 16,
    // this is the width for the product
    // parameter PRODUCT_WIDTH = 8,
    // this is the width for the summed product
    parameter OUT_WIDTH = IN_WIDTH + W_WIDTH + $clog2(IN_SIZE),

    // this defines the number of elements in the vector, this is tunable
    // when block arithmetics are applied, this is the same as the block size
    parameter IN_SIZE = 4

) (
    input clk,
    input rst,

    // input port for activations
    input  logic [IN_WIDTH-1:0] data_in      [IN_SIZE-1:0],
    input                       data_in_valid,
    output                      data_in_ready,

    // input port for weights
    input  logic [W_WIDTH-1:0] weights      [IN_SIZE-1:0],
    input                      weights_valid,
    output                     weights_ready,

    // output port
    output logic [OUT_WIDTH-1:0] data_out,
    output                       data_out_valid,
    input                        data_out_ready

);

  localparam PRODUCT_WIDTH = IN_WIDTH + W_WIDTH;


  logic [PRODUCT_WIDTH-1:0] pv       [IN_SIZE-1:0];
  logic                     pv_valid;
  logic                     pv_ready;
  fixed_vector_mult #(
      .IN_WIDTH(IN_WIDTH),
      .W_WIDTH (W_WIDTH),
      .IN_SIZE (IN_SIZE)
  ) fixed_vector_mult_inst (
      .clk(clk),
      .rst(rst),
      .data_in(data_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .weights(weights),
      .weights_valid(weights_valid),
      .weights_ready(weights_ready),
      .data_out(pv),
      .data_out_valid(pv_valid),
      .data_out_ready(pv_ready)
  );


  // sum the products
  logic [OUT_WIDTH-1:0] sum;
  logic                 sum_valid;
  logic                 sum_ready;
  // sum = sum(pv)
  fixed_adder_tree #(
      .IN_SIZE (IN_SIZE),
      .IN_WIDTH(PRODUCT_WIDTH)
  ) fixed_adder_tree_inst (
      .clk(clk),
      .rst(rst),
      .data_in(pv),
      .data_in_valid(pv_valid),
      .data_in_ready(pv_ready),

      .data_out(sum),
      .data_out_valid(sum_valid),
      .data_out_ready(sum_ready)
  );

  // Picking the end of the buffer, wire them to the output port
  assign data_out = sum;
  assign data_out_valid = sum_valid;
  assign sum_ready = data_out_ready;

endmodule
