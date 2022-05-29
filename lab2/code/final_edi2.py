import re
import time
#原子公式
class Formula():
    def __init__(self, flag, predicate, variable):
        self.flag = flag #表示一个原子公式中的是或非，1或0
        self.predicate = predicate #表示原子公式中的谓词名称，字符串
        self.variable = variable #表示原子公式中的变元或常量，列表

    def print(self):
        if self.flag == 0:
            print("!", end="")
        print(self.predicate, "(", sep="", end="")
        for i in range(len(self.variable)):
            if i != 0:
                print(",", end="")
            print(self.variable[i], end="")
        print(")", end="")

    def __eq__(self, other):
        if (self.flag == other.flag) and (self.predicate == other.predicate) and (self.variable == other.variable):
            return True
        else:
            return False


#子句
class Clause():
    def __init__(self, formulas, number=-1, left=None, right=None):
        self.formulas = formulas #子句中的所有原子公式，列表
        self.left = left #左子树，构建输出时使用
        self.right = right #右子树
        self.number = number #该子句的序号，如果为前提条件，则序号为在子句集中的位置；如果是后续归结出的子句，则序号标注为-1

    def print(self):
        if len(self.formulas) == 0:
            print("[]")
            return
        if len(self.formulas) == 1:
            self.formulas[0].print()
            print()
        else:
            for i in range(len(self.formulas)):
                if i != 0:
                    print(",", end="")
                self.formulas[i].print()
            print()


#将字符串转换为Formula类的对象，并返回该对象
def build_formula(formula_str):
    flag = 1  #标志是非
    #如果是非开头
    if formula_str[0] == "!":
        formula_str = formula_str.replace("!", "") #化为统一形式
        flag = 0
    #分割得到原子公式的各个部分
    formula_list = re.split(r'(?:[(),])', formula_str)#优于split函数
    variable = []
    for i in range(1, len(formula_list) - 1): #去掉因为反括号结尾而形成的空列表项
        variable.append(formula_list[i])
    return Formula(flag, formula_list[0], variable)

def get_condition(s):#处理字符串
    n = input()
    for i in range(int(n)):
        clause_str = input() #输入一个条件子句
        clause_formulas = [] #用于储存一个子句中的原子公式
        #如果以"("开头说明该子句中有多个原子公式
        if(clause_str[0] == "("):
            clause_str = clause_str[1:-1] #去掉字符串前后的括号
            formulas = clause_str.split(", ") #分割得到原子公式
            for j in range(len(formulas)):
                temp = build_formula(formulas[j]) #构建Formula对象
                #temp.print()
                clause_formulas.append(temp)
        #子句即为原子公式
        else:
            clause_formulas.append(build_formula(clause_str))
        clause = Clause(clause_formulas, len(s))
        s.append(clause)
    return n

#拷贝子句，并返回副本
def copy_clause(clause):
    new_formulas = []
    for i in range(len(clause.formulas)):
        new_formula = Formula(clause.formulas[i].flag, clause.formulas[i].predicate, clause.formulas[i].variable[:])
        new_formulas.append(new_formula)
    new_clause = Clause(new_formulas, clause.number)
    return new_clause

#改变子句中指定的变量为指定的常量
def change_variable(clause, before, after):
    for i in range(len(clause.formulas)):
        for j in range(len(clause.formulas[i].variable)):
            if clause.formulas[i].variable[j] == before:
                clause.formulas[i].variable[j] = after

#将抵消的原子公式去除，合并剩下的原子公式，并返回原子公式列表
def get_new_formulas(formula1, formula2, clause1_c, clause2_c):
    new_formulas = []
    for i in range(len(clause1_c.formulas)):
        #如果不是抵消掉的原子公式
        if clause1_c.formulas[i] is not formula1:
            #如果这个原子公式没有在新的原子公式集合中出现过，则放入新的原子公式集合
            if clause1_c.formulas[i] not in new_formulas:
                new_formulas.append(clause1_c.formulas[i])
    for i in range(len(clause2_c.formulas)):
        if clause2_c.formulas[i] is not formula2:
            if clause2_c.formulas[i] not in new_formulas:
                new_formulas.append(clause2_c.formulas[i])
    return new_formulas

#对两个子句进行归结合一
def unify(clause1, clause2, s):
    #为了保存子句的原始信息用于最后的输出，先将子句进行拷贝
    clause1_c = copy_clause(clause1)
    clause2_c = copy_clause(clause2)
    #在两个子句中找是否有可以匹配的原子公式
    for i in range(len(clause1_c.formulas)):
        formula1 = clause1_c.formulas[i]
        for j in range(len(clause2_c.formulas)):
            formula2 = clause2_c.formulas[j]
            #如果是非相反，谓词相同
            if (formula1.flag != formula2.flag) and (formula1.predicate == formula2.predicate):
                for k in range(len(formula1.variable)):
                    #如果第一个是变量，第二个是常量，则对第一个子句进行合一
                    if (len(formula1.variable[k]) == 1) and (len(formula2.variable[k]) != 1):
                        change_variable(clause1_c, formula1.variable[k], formula2.variable[k])
                    #如果第一个是常量，第二个是变量，则对第二个子句进行合一
                    if (len(formula1.variable[k]) != 1) and (len(formula2.variable[k]) == 1):
                        change_variable(clause2_c, formula2.variable[k], formula1.variable[k])
                #如果可以归结，也就是常量都相等
                if (formula1.variable == formula2.variable):
                    #得到新的原子公式集合
                    new_formulas = get_new_formulas(formula1, formula2, clause1_c, clause2_c)
                    #构建新的子句，并记录薪子句是从哪两个原子公式归结来的
                    new_clause = Clause(formulas=new_formulas, left=clause1, right=clause2)
                    s.append(new_clause)
                    #如果新的子句为空集则归结完成
                    if len(new_clause.formulas) == 0:
                        return 1
                    else:
                        return 0

#将子句集中的子句两两进行比较，并进行归结合一
def matching(s, start_clause):
    n = len(s) #先暂时不考虑在归结过程中产生的新的子句
    for i in range(n):
        clause1 = s[i]
        for j in range(max(i + 1, start_clause), n):
            clause2 = s[j]
            #对两个子句进行归结合一
            end = unify(clause1, clause2, s)
            #如果归结结束
            if end == 1:
                return 1
    return 0

#根据以空集为根的树反推出形成空集所需要的过程子句，并放入一个栈中
def get_stack(s):
    stack = []
    #首先将根放入队列
    queue = [s.pop()]
    #层次遍历以空子句为根的树
    while queue:
        root = queue.pop(0) #取队首元素作为新的根节点
        stack.append(root) #将结点放入栈中
        #root.print()
        #将左右子节点放入栈中
        if (root.left != None) and (root.right != None):
            queue.append(root.left)
            queue.append(root.right)
    return stack

#得到合一过程中变量的替换信息
def get_chg_v(clause1, clause2):
    v = []
    for i in range(len(clause1.formulas)):
        formula1 = clause1.formulas[i]
        for j in range(len(clause2.formulas)):
            formula2 = clause2.formulas[j]
            if (formula1.flag != formula2.flag) and (formula1.predicate == formula2.predicate):
                for k in range(len(formula1.variable)):
                    # 如果第一个是变量，第二个是常量
                    if (len(formula1.variable[k]) == 1) and (len(formula2.variable[k]) != 1):
                        v.append(formula1.variable[k])
                        v.append(formula2.variable[k])
                    # 如果第一个是常量，第二个是变量
                    if (len(formula1.variable[k]) != 1) and (len(formula2.variable[k]) == 1):
                        v.append(formula2.variable[k])
                        v.append(formula1.variable[k])
                return v

#得到能够抵消的两个原子公式在子句中的位置
def get_f_index(clause1, clause2):
    for i in range(len(clause1.formulas)):
        formula1 = clause1.formulas[i]
        for j in range(len(clause2.formulas)):
            formula2 = clause2.formulas[j]
            if (formula1.predicate == formula2.predicate) and (formula1.flag != formula2.flag):
                #如果子句中只包含一个原子公式，则不需要定位
                if len(clause1.formulas) == 1:
                    f1_inx = -1
                else:
                    f1_inx = i
                if len(clause2.formulas) == 1:
                    f2_index = -1
                else:
                    f2_index = j
                return f1_inx, f2_index

#根据定位的int转为对应的字母
def chg_to_str(f1_inx, f2_inx):
    index = {0:"a", 1:"b", 2:"c", -1:""}
    return index[f1_inx], index[f2_inx]

def print_str(number1, f1_str, number2, f2_str, v, next_clause):
    print("R[", number1 + 1, f1_str, ",", number2 + 1, f2_str, "]", sep="", end="")
    if len(v) == 2:
        print("(", v[0], "=", v[1], ")", sep="", end="")
    elif len(v) == 4:
        print("(", v[0], "=", v[1], ",", v[2], "=", v[3], ")", sep="", end="")
    print(" = ",sep="", end="")
    next_clause.print()

#得到归结后的子句
def get_next(stack, clause1, clause2):
    for i in range(len(stack)):
        if (stack[i].left != None) and (stack[i].right != None):
            #如果左子树和右子树分别为clause2和clause1说明该子句是由clause1和clause2归结而成
            if (stack[i].left is clause2) and (stack[i].right is clause1):
                return stack[i]

#输出过程信息
def print_result(stack):
    index1 = len(stack) - 1
    index2 = len(stack) - 2
    while index2 > 0:
        clause1 = stack[index1]
        clause2 = stack[index2]
        # 得到能够抵消的两个原子公式在子句中的位置
        f1_inx, f2_inx = get_f_index(clause1, clause2)
        #将位置信息转化为字母
        f1_str, f2_str = chg_to_str(f1_inx, f2_inx)
        #得到合一过程中变量替换的信息
        v = get_chg_v(clause1, clause2)
        #得到归结后的子句
        next_clause = get_next(stack, clause1, clause2)
        #按照格式输出
        print_str(clause1.number, f1_str, clause2.number, f2_str, v, next_clause)
        index1 -= 2
        index2 -= 2

#得到输出时子句对应的序号
def chg_number(stack, count):
    #先将栈中的元素调换顺序，变成结果在最后，这样在列表中的位置对应的就是子句的序号
    stack.reverse()
    for clause in stack:
        #如果是在过程中新生成的子句
        if clause.number == -1:
            clause.number = count
            count += 1
    #还原为栈的顺序
    stack.reverse()

def ui():
    print('----')
    print('--------命题逻辑归结推理系统--------')
    print('----')


def main():
    s = [] #子句集
    ui()
    count = get_condition(s) #将前提条件封装，返回条件子句个数
    # for i in range(len(s)):
    #     s[i].print()
    start = time.time()
    start_clause = 0 #表示比较开始的子句位置
    end = 0 #标志是否出现空子句
    while end == 0:
        start_clause_temp = len(s) #下一次循环的两两比较从这里开始，就不用每次都从头开始了
        end = matching(s, start_clause) #归结合一
        start_clause = start_clause_temp #更新比较开始位置
    #根据空子句的Clause树中的位置反推合成过程中所需要的子句，并储存到栈中
    stack = get_stack(s)
    # for i in stack:
    #     i.print()
    chg_number(stack, int(count)) #根据需要的子句改变其对应的序号
    print_result(stack) #按照格式输出结果
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()