#include <stdio.h>

int main(void) {
  
  int B[11]={0}, i, a=0;
  
  printf("wpisz liczby: ");

  while(a!=-1){
  i = scanf("%d", &a);
  if(i==0){
      printf("Incorrect input");
      return 1;
    }
  if(a>=0 && a<=10)
  B[a]++;
 }

  for(i=0; i<11; i++)
  printf("%d - %d\n", i, B[i]);

  return 0;
}