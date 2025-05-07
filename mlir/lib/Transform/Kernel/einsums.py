import opt_einsum as oe
import os


def main(input_filename=None,
         output_filename="optimized.einsum",
         overwrite=False,
         enable_cotengra=False):
    if input_filename is None:
        raise ValueError("Must supply a filename")

    f = open(input_filename, "r")
    spec = f.readline().replace("\n", "")
    output = spec.split("->")[1]

    lines = [line.split(":") for line in f]
    inputs_to_shapes = {
        line[0]: tuple(int(ax_len) for ax_len in
            line[1].replace("(", "").replace(")", "").strip("\n").split(","))
        for line in lines
        if line[0] != output
    }

    if enable_cotengra:
        opt = ct.HyperOptimizer()
    else:
        opt = "greedy"

    opt_expression = oe.contract_expression(
        spec, *list(inputs_to_shapes.values()), optimize=opt)

    new_specs = ""
    for contraction in opt_expression.contraction_list[:-1]:
        new_specs += (contraction[2] + "\n")
    new_specs += opt_expression.contraction_list[-1][2]

    if os.path.exists(output_filename) and not overwrite:
        raise ValueError(
            f"`{output_filename}` exists. Use a different filename or run "
            "this script again with `-f`")

    with open(output_filename, 'w') as f:
        print(new_specs, file=f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-filename",
                        action="store",
                        help="File containing einsum specification information")
    parser.add_argument("-o", "--output-filename",
                        action="store",
                        help="Where to write the optimized specification."
                        "Default is `optimized.einsum`",
                        default="optimized.einsum")
    parser.add_argument("-f", "--force",
                        action="store_true",
                        help="Overwrite existing output file. Default output "
                        "filename is `optimized.einsum`.")
    parser.add_argument("-c", "--enable-cotengra",
                        action="store_true",
                        help="Use cotengra to optimize the einsum expression")

    args = parser.parse_args()

    if args.enable_cotengra:
        try:
            import cotengra as ct
        except ImportError:
            raise ValueError(
                "Cotengra is not installed. Either install cotengra, or run "
                "this script again without -enable-cotengra")

    main(args.input_filename,
         args.output_filename,
         args.force,
         args.enable_cotengra)
