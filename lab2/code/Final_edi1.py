import copy
import time
S = [] # 放入每一行的数据，例如第一行是[{'On': ['aa', 'bb'], 'posnum': 1}]
V={}#替换值
M={}
M[0]='a' #为了输出R[]中的内容
M[1]='b'
M[2]='c'
M[3]='d'
M[4]='e'
cot=1
def GetClauseSet(filePath):#这个函数只处理最后一行
    global S #字典，用来记录每一行的谓词和变量
    global cot 
    for line in open(filePath,encoding = 'utf-8'):
        line = line.replace(' ', '').strip() #先把空格和不可见字符（如换行）去掉
        #print(line)
        if line[0]=='(': #如果是最后一行
           line=list(line) #把字符串变成了list，好操作
           line[0]='' #去掉头尾两个括号
           line[len(line)-1]=''
           line=''.join(line) #因为是''.join所以相当于加起来
        for i in range(len(line)-2):#因为替换了两个头尾，所以缩短
            if line[i+1]==','and line[i]==')':#这里也是处理最后一行，为了把多个函数分开
                line=list(line) 
                line[i+1]=';' #改变符号是为了后面进行split
                line=''.join(line) 
                #print(line)
        line = line.split(';') #通过;来分割字符串,每个分到元组元素中,为了和前几行做区分使用了;
        #print(line)
        newele={}
        for i in range(len(line)): #
            str1=line[i] #每次读入的元素
            for j in range(len(str1)-1):
                str2=str1#为了不改变str1的值，做一个保留
                if str1[j]=='(':
                   elename=str2[0:j] #找到了函数名称
                   elemem=str2[j+1:len(str1)-1] #用到了str1,elemem是剩下的字符串
                   elemem=elemem.split(',')#分割开
                   newele[elename]=elemem#填充字典，比如'On': ['aa', 'bb']
                   line[i]=newele
                   break
        newele['posnum']=cot #这个参数对应行数
        #例如第一行执行完之后newele里面是{'On': ['aa', 'bb'], 'posnum': 1}
        cot+=1 #记录行数
        S.append(newele) #可以将S认为是二维字典元组，S最后应该是[{'On': ['aa', 'bb'], 'posnum': 1}, {'On': ['bb', 'cc'], 'posnum': 2}, {'Green': ['aa'], 'posnum': 3}, {'!Green': ['cc'], 'posnum': 4}, {'!On': ['x', 'y'], '!Green': ['x'], 'Green': ['y'], 'posnum': 5}]
        #print("\n S here")
        #print(S)

def opposite(clause):#判断是否有对立语句可以进行抵消
    if '~' in clause:
        return clause.replace('~', '')
    else:
        return '~' + clause

def unify():
    global S
    global cot
    end = False
    while True:
        if end: break 
        father = S.pop()
        jflag=False #判断函数的可合成性
        for i,val in father.items(): #i是索引，val是里面的值
            if end: break
            for mother in S:
                #print("mother",mother)
                if end: break
                j=[]
                pos2=''
                pos3=''
                for key,value in mother.items(): #一个意思，key是索引，value是里面的值
                    if key==opposite(i):#大前提
                        dif=0#判断不同的谓词有多少个
                        for pos1 in value:#mother里面的变量
                            if pos1 in val:
                                continue
                            else:
                                dif+=1
                                pos2=pos1#记录不同的谓词
                        if dif<=1:#说明可以进行合成
                            for h in range(len(val)):#一种思路是找变量x,y,z，最开始的想法，但是会全盘替换，难以实现，此思路为直接判断不同的谓词有几个，确实有问题
                                if val[h] in value:#存在一样的就跳过
                                    continue
                                else:
                                    pos3=val[h]#记录father和mother里面不同的那个变量
                                    #print("pos3",pos3)
                            flag=True#说明可以合成
                            #print("V",V)
                            for key1 in V.values():#如果这个函数已经被替换过了，就不可再替换，为了修正82行的错误
                                if pos2 ==key1:
                                    flag=False
                                    break
                            if flag==True:
                                j.append(key) #记录每次取到的可替换函数
                                break
                if j == []:
                    continue
                else:
                    jflag=True#j不是空的，修改标志位为可替换
                    #print('R[',father['posnum'],',',mother['posnum'],'] = ',end=' ')
                    print('R[',end='')
                    m=0#为了确定输出里面的字母
                    for k1 in father.keys():
                        if k1==i:
                            break
                        m+=1
                    n=0#同理
                    for k2 in mother.keys():
                        if k2==key:
                            break
                        n+=1
                    print(father['posnum'],end='')
                    if len(father)>2:#意思是可以进行合成
                        print(M[m],end='')
                    print(',',end='')
                    print(mother['posnum'],end='')
                    if len(mother)>2:#同理
                        print(M[n],end='')
                    print(']',end='')#R到此输出结束
                    if pos2!='' and pos3!='':
                        print('(',end='')
                        print(pos3,'=',pos2,end='')
                        print(') ',end='')
                    print('= ',end='')
                    newele1={}
                    newele2={}
                    newele1=copy.deepcopy(father)
                    newele2=copy.deepcopy(mother)
                    if pos2!='' and pos3!='':   
                        V[pos2]=pos3
                        for vk,v in newele1.items():
                            if vk=='posnum':
                                break
                            for k in range(len(v)):
                                if v[k]==pos3:
                                    v[k]=pos2
                    del newele1[i]
                    del newele2[j[0]]
                    if len(newele1)>1:#判断是否需要加行数，并重新构造一个新的行
                        newele1['posnum']=cot
                        cot+=1
                        S.append(newele1) 
                    if len(newele2)>1:#同上
                        newele2['posnum']=cot
                        cot+=1
                        S.append(newele2)
                    S.remove(mother)#与命题逻辑一致，mother使用完毕
                    if len(newele1) == 1 and len(newele2) == 1:
                        print('[]')
                        end = True
                    elif len(newele1) == 1:
                        cot2=0 
                        for key,value in newele2.items():
                            if key=='posnum':
                                break
                            print(key,'(',end='')
                            for m in range(len(value)):
                                print(value[m],end='')
                                if m!=len(value)-1:
                                    print(',',end='')
                            print(')',end='')
                            if cot2!=len(newele2)-2:
                                print(',',end='')
                            cot2+=1
                        print('\n',end='')
                    elif len(newele2) == 1:
                        cot2=0 
                        for key,value in newele1.items():
                            if key=='posnum':
                                break
                            print(key,'(',end='')
                            for m in range(len(value)):
                                print(value[m],end='')
                                if m!=len(value)-1:
                                    print(',',end='')
                            print(')',end='')
                            if cot2!=len(newele1)-2:
                                print(',',end='')
                            cot2+=1
                        print('\n',end='')
                    else:
                        cot2=0 
                        cot3=0
                        for key,value in newele1.items():
                            if key=='posnum':
                                break
                            print(key,'(',end='')
                            for m in range(len(value)):
                                print(value[m],end='')
                                if m!=len(value)-1:
                                    print(',',end='')
                            print(')',end='')
                            if cot2!=len(newele1)-2:
                                print(',',end='')
                            cot2+=1
                        print(' and',end=' ')
                        for key,value in newele2.items():
                            if key=='posnum':
                                break
                            print(key,'(',end='')
                            for m in range(len(value)):
                                print(value[m],end='')
                                if m!=len(value)-1:
                                    print(',',end='')
                            print(')',end='')
                            if cot3!=len(newele2)-2:
                                print(',',end='')
                            cot3+=1
                        print('\n',end='')
        if jflag==False:
            S.insert(0,father)

def ui():
    print('----')
    print('--------命题逻辑归结推理系统--------')
    print('----')

def main():
    start = time.time()
    filePath = r'input2.txt'
    GetClauseSet(filePath)
    ui()
    print(len(S))
    with open('input2.txt') as file_object:
        contents = file_object.read()
        print(contents)
    unify()
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()