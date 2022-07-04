#include <stdio.h>

int main(void)
{   
	int a,b,c,n,m;
	printf("podaj liczbe 1\n");
	scanf("%d %d %d",&a,&b,&c);

	 n=(a>=b);
     m=(a>=c);
	printf("%d",n+m);
	return 0;
}