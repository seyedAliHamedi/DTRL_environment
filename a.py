def main():
    a=0.9
    b=0.9995
    c=0
    while a>0.50:
        a*=b
        c+=1
    print(c)
    while a>0.1:
        a*=b
        c+=1
    print(c)
main()