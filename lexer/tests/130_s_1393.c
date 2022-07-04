#include <stdio.h>
#include <stdlib.h>

int main()
{
    int imputSeconds;
    int sec,min,hur;

    printf("Input number of seconds: ");
    scanf("%d",&imputSeconds);

    sec = imputSeconds % 60;
    imputSeconds -= sec;
    imputSeconds /= 60;

    min = imputSeconds % 60;
    imputSeconds -= min;
    imputSeconds /= 60;

    hur = imputSeconds % 60;

    printf("%02d:%02d:%02d",hur,min,sec);

    return 0;
}
