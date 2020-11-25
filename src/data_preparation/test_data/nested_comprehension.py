empty_list = []
generated_list = [i for i in range(10)]

single_nested_list = [[i * j for j in range(i)] for i in range(10)]
double_nested_list = [[[i * j * k for k in range(j)] for j in range(i)] for i in range(10)]

single_comprehended_list = [i * j for i in range(10) for j in range(i)]
double_comprehended_list = [i * j * k for i in range(10) for j in range(i) for k in range(j)]
