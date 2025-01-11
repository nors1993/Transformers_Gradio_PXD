import json
labels_0 = []
with open(r"datasets\medical_test_datasets_light.csv",'r',encoding="utf-8") as file:
    # file.readlines()[1:] 表示从第二行开始读取
    # 删除前两个逗号分隔的之前的全部内容
    for line in file.readlines()[1:]:
        line = line.strip()
        first_comma = line.find(",")
        second_comma = line.find(",", first_comma + 1)
        last_comma = line.rfind(",")  # 从右侧开始找到的第一个逗号
        # 找到前两个逗号和最后一个逗号
        if first_comma != -1 and second_comma != -1 and last_comma != -1:
            # 拼接删除内容后的行
            new_line = line[second_comma + 1:last_comma]
        else:
            new_line = line  # 如果没有两个逗号，则保留原来的行
        
        new_line = new_line[1:-1]  # 提取首尾双引号之间的内容
        new_line = new_line.replace('""', '"')  # 删除多余的双引号，即使用"将""替换
        new_line = new_line.replace("\\n\\n", '')
        new_line = new_line.replace('[', '')
        new_line = new_line.replace(']', '')
        
        #print (new_line)
        
        labels_0.append(new_line)
    
        #print(type(labels_0),labels_0)  # lables_0为list ['{},{},...'],元素为str类型
    labels_1 = labels_0  # list
    #print(type(labels_1), labels_1)

    items = ""
    for item in labels_1:
        items += item + ','
        #print(type(items), items)
    items = items[0:-1]  #删除最后一个元素的末尾逗号
    #print(type(items), items)  # str

    # 将字符串 items 转换为有效的 JSON 格式
    json_string = f'[{items}]'  # 将多个对象放入列表中
 
    labels_2 = json.loads(json_string) # 转换为字典列表
    #print(type(labels_2), labels_2)

    # 提取列表中的字典的键是human或gpt对应的value值并配对
    human_values = []
    gpt_values = []
    for item in labels_2:
        if item["from"] == "human":
            human_values.append(item["value"])
        elif item["from"] == "gpt":
            gpt_values.append(item["value"])
    #print(human_values)
    # 按顺序组合 human 和 gpt 的值
    paired_values = []
    for Q, A in zip(human_values, gpt_values):
        paired_values.append({"human": Q, "gpt": A})
    #print(type(paired_values), paired_values)


    out_jsonl = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }]}
    
    for Q, A in zip(human_values, gpt_values):
           out_jsonl["messages"].append(
               {
                "role": "user",
                "content": Q
            }             
           )
           out_jsonl["messages"].append(
               {
                "role": "assistant",
                "content": A
            }             
           )
# 指定输出文件名
output_file = r'FT\val_datasets_generated.jsonl'

# 写入 .jsonl 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json_line = json.dumps(out_jsonl, ensure_ascii=False)
    f.write(json_line + '\n')

print(f"内容已写入 {output_file} 文件。")







   
   
            
  


