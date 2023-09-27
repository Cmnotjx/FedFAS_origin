# x = torch.load(temp_proto_path+'glo_proto.pt')
            # print(len(x))
            # print(len(x[0]))
            # a = x[0]
            # b = x[1]
            # print(len(a[0]))
            # print(len(b[0]))
            # it_num = 0
            # no_num = 0
            # for i, j in zip(a[3],b[3]):    # protos use is protos not glb_proto
            #     # print(i.shape,j.shape)
            #     if i.shape==j.shape:
            #     # if i.shape==j.shape and i.shape!=torch.Size([1,256]):
            #         # print(i.shape)
            #         it_num+=1
            #     elif i.shape!=j.shape:
            #         no_num+=1
            # print(it_num,no_num)
            # return 1

# if _config['eval_fold'] == 0:
#     if name == 'CHAOST2_Superpix_1':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_2':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_3':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_4':
#         dy_rounds = i + 1
# elif _config['eval_fold'] == 1:
#     if name == 'CHAOST2_Superpix_1':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_2':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_3':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_4':
#         dy_rounds = i + 1
# elif _config['eval_fold'] == 2:
#     if name == 'CHAOST2_Superpix_1':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_2':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_3':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_4':
#         dy_rounds = rounds
# elif _config['eval_fold'] == 3:
#     if name == 'CHAOST2_Superpix_1':
#         dy_rounds = rounds
#     elif name == 'CHAOST2_Superpix_2':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_3':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_4':
#         dy_rounds = i + 1
# elif _config['eval_fold'] == 4:
#     if name == 'CHAOST2_Superpix_1':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_2':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_3':
#         dy_rounds = i + 1
#     elif name == 'CHAOST2_Superpix_4':
#         dy_rounds = rounds