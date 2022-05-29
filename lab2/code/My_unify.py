S = [] #存储子句

def read_file(filePath):
    global S
    for line in open(filePath,mode = 'r',encoding='utf-8'):
        line = line.replace(' ','').strip()
        line = line.split(',')
        S.append(line)

def opposite(line):
    if '~' in line:
        return line.replace('~','')
    else:
        return '~' + line

def unify():
    global S
    end = False
    while True:
        if end: break
        father = S.pop()
        for i in father[:]:
            if end: break
            for mother in S[:]:
                if end: break
                j = list(filter(lambda x: x==opposite(i),mother))
                if j == []:
                    continue
                else:
                    print('\n亲本子句：' + ' , '.join(father) + ' 和 ' + ' , '.join(mother))
                    father.remove(i)
                    mother.remove(j[0])
                    if(father == [] and mother == []):
                        print('归结式：NIL')
                        end = True
                    elif father == []:
                        print('归结式：' + ' , '.join(mother))
                    elif mother == []:
                        print('归结式：' + ' , '.join(mother))
                    else:
                        print('归结式：' + ' , '.join(father) + ' , ' + ' , '.join(mother))

def ui():
    print('----')
    print('--------命题逻辑归结推理系统--------')
    print('----')


def main():
    filePath = r'input5.txt'
    read_file(filePath)
    ui()
    unify()


if __name__ == '__main__':
    main()
