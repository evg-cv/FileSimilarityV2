def check_text(str_list):
    checked_list = []
    for s_list in str_list:
        if type(s_list) is str:
            checked_list.append(s_list)

    return checked_list


if __name__ == '__main__':
    check_text(str_list=[])
