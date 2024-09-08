def main():
    a=0.9
    b=0.999
    c=0
    while a>0.01:
        a*=b
        c+=1
    print(c)
main()