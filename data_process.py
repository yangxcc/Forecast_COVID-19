import json, csv, requests  # 导入请求模块

def get_data():  # 定义获取数据并写入csv文件里的函数
    headers = {"user-agent": "Mizilla/5.0"}
    url = "https://c.m.163.com/ug/api/wuhan/app/data/list-total"
    response = requests.get(url, headers=headers)  # 发出请求并json化处理
    # print(response) #测试一下是否获取数据了
    data = json.loads(response.text)  # 提取数据部分
    list_datas_1 = data['data']['chinaDayList']
    # print(list_datas_1)
    # print(data.keys()) # 获取数据组成部分['chinaTotal', 'chinaAdd', 'lastUpdateTime', 'areaTree', 'chinaDayList']
    update_time = data['data']["lastUpdateTime"]  # 更新时间
    chinaDayList = data['data']["chinaDayList"]  # 历史数据
    # print(chinaDayList)
    with open("data/疫情单日新增.csv", "w+", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["date", "confirm", "suspect", "dead", "heal", "update_time"]  # 定义表头
        writer.writerow(header)
        for i in range(len(chinaDayList)):
            data_row1 = [chinaDayList[i]["date"], chinaDayList[i]['today']["confirm"], chinaDayList[i]['today']["suspect"],
                         chinaDayList[i]['today']["dead"], chinaDayList[i]['today']["heal"], update_time]
            writer.writerow(data_row1)

    with open("data/疫情累计人数.csv","w+",newline="") as csv_file:
        writer=csv.writer(csv_file)
        header = ["date", "confirm", "suspect", "dead", "heal", "update_time"]  # 定义表头
        writer.writerow(header)
        for i in range(len(chinaDayList)):
            data_row1 = [chinaDayList[i]["date"], chinaDayList[i]['total']["confirm"],
                         chinaDayList[i]['total']["suspect"],
                         chinaDayList[i]['total']["dead"], chinaDayList[i]['total']["heal"], update_time]
            writer.writerow(data_row1)

    # areaTree = data['data']["areaTree"]  # 各地方数据
    #     # print(areaTree)

if __name__ == "__main__":
    get_data()