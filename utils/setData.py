import os

if __name__ == "__main__":
    prompts = []
    mesh_dir = "data/mesh/"
    config_dir = "configs/"
    config_example = "pineapple_appearance.json"
    f = open(mesh_dir + "prompt.txt", 'r')
    lines = f.readlines()

    for line in lines:
        prompts.append(line.strip())
    # print(prompts)
    f.close()


    f1 = open(config_dir+config_example, 'r')
    lines = f1.readlines()
    f1.close()

    for prompt in prompts:
        f2 = open(config_dir+prompt+".json", 'w')
        new_file=[]
        for line in lines:
            token = line.strip().split(':')
            if len(token) == 2:
                print(token[1])
                if token[1] == ' "out/pineapple/dmtet_mesh/mesh.obj",': # base mesh
                    new_file.append(token[0] + ': "' + mesh_dir + prompt + '/mesh.obj",\n')
                elif token[1] == ' "A pineapple",': # prompt
                    new_file.append(token[0] + ': "' + prompt + '",\n')
                elif token[1] == ' "pineapple_appearance_0",': # out dir
                    new_file.append(token[0] + ': "' + prompt + '",\n')
                else:
                    new_file.append(token[0] + ":" + token[1] + "\n")
            else:
                new_file.append(line)

        f2.write(''.join(new_file))
        f2.close()   
