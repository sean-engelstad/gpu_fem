def read_gmres_file_and_print_inline_lists(filename):
    iterations = []
    residuals = []
    
    with open(filename, 'r') as file:
        for line in file:
            if "GMRES iter" in line:
                parts = line.strip().split()
                iterations.append(int(parts[2]))
                residuals.append(float(parts[-1]))
            if "GMRES[" in line:
                parts = line.strip().split('[')
                # print(f"{parts=}")
                parts2 = parts[1].strip().split(']')
                part2 = parts2[0]
                part3 = parts2[-1].split(":")[-1].strip()
                # print(f'{parts=} {part2=} {part3=}')
                iterations.append(int(part2))
                residuals.append(float(part3))
                

    # Print both lists on one line each
    iter_str = "iter=[" + ", ".join(str(i) for i in iterations) + "],"
    resid_str = "resids=[" + ", ".join(f"{r:.15e}" for r in residuals) + "],"
    
    print("gmres_list += [GMRES(")
    print("fill= ,")
    print(iter_str)
    print(resid_str)
    print(")]")

# Example usage
read_gmres_file_and_print_inline_lists("gmres_in.txt")
