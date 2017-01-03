#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#define MAX_CODE_LENGTH 40

long long vocab_max_size = 1000, 
     vocab_size = 0, 
     layer1_size = 100;


/**
 * word与Huffman树编码
 */
struct vocab_word {
  long long cn;     // 词在训练集中出现的频率
  int *point;       // 编码的节点路径
  char *word,       // 词
       *code,       // Huffman编码，每一位上，0或1
       codelen;     // Huffman编码长度
};


struct vocab_word *vocab;

/*
 * 打印构造过程的中间状态.
 */ 
void printState(long long* count, 
            long long* binary,
            long long* parent_node) {
  printf("count[]:\t");
  for(int x=0; x<vocab_size * 2; x++) {
    printf("%lld", count[x]);
    printf(" ");
  }
  printf("\n");

  printf("binary[]:\t");
  for(int x=0; x<vocab_size * 2; x++) {
    printf("%lld", binary[x]);
    printf(" ");
  }
  printf("\n");

  printf("parent[]:\t");
  for(int x=0; x<vocab_size * 2; x++) {
    printf("%lld", parent_node[x]);
    printf(" ");
  }
  printf("\n");

}


/**
 * 使用词频创建一棵的Huffman树. 频率高的字将具有short唯一
 * 的二进制码(binary code).
 *
 */
// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, 
       b, 
       i, 
       min1i, 
       min2i, 
       pos1, 
       pos2, 
       point[MAX_CODE_LENGTH];


  char code[MAX_CODE_LENGTH];


  // count: 词频.
  // binary:
  // parent_node: 
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

  // 初始化count数组的前一半
  for (a = 0; a < vocab_size; a++) {
      count[a] = vocab[a].cn;
      //printf("count=%lld\n", count[a]);
  }

  // 初始化count数组的后一半，用于交换. 赋很大值.
  for (a = vocab_size; a < vocab_size * 2; a++) {
      count[a] = 1e15;
      //printf("count=%lld\n", count[a]);
  }

  // 
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  //printf("pos1=%lld, pos2=%lld\n", pos1, pos2);

  printState(count, binary, parent_node);  

  // 根据算法构建Huffman树，一次增加一个节点.
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {

    printf("----------------\n");
    printf("pos1=%lld, pos2=%lld\n", pos1, pos2);

    // 每轮找到最小的两个值.
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {

      // 遍历所有词汇的count，比较count；取较小值.
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {  
      min1i = pos2;
      pos2++;
    }

    printf("min1i=%d, min2i=%d\n", min1i, min2i);
    printf("pos1=%lld, pos2=%lld\n", pos1, pos2);
    // 再比一次.
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }

    // 最小值cnt的两个索引
    printf("min1i=%d, min2i=%d\n", min1i, min2i);
    printf("count[min1i]=%d, count[min2i]=%d\n", count[min1i], count[min2i]);

    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;

    printf("count[vocab_size + a] = %d\n", count[vocab_size + a]);
    printf("parent_node[%d] = %d\n", min1i, parent_node[min1i]);
    printf("parent_node[%d] = %d\n", min2i, parent_node[min2i]);
    printf("binary[%d] = %d\n", min2i, binary[min2i]);
    
    printState(count, binary, parent_node);  

  }

  // 将二进制编码分配给词汇表中每个词汇.
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;

    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }

    // 得到huffman编码长度.
    vocab[a].codelen = i;

    // 得到huffman编码code及路径point.
    vocab[a].point[0] = vocab_size - 2;
    
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }

  // 释放内存.
  free(count);
  free(binary);
  free(parent_node);
}

/**
 * 
 */
int main()
{
    vocab_size = 6; 
    vocab = (vocab_word*) calloc(vocab_size+1, sizeof(vocab_word));
    memset(vocab, 0, sizeof(vocab_word) * vocab_size+1);

    // 初始化code/point.
    for (int a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }

    // 事先对vocab按词频排好序(word2vec事先已经用qsort处理)，从大到小排序.
    // 可以用qsort。 本代码直接已经人工排好序了.
    vocab[0].cn = 7;
    char* str = "T";
    vocab[0].word = str;

    vocab[1].cn = 5;
    str = "E";
    vocab[1].word = str;

    vocab[2].cn = 4;
    str = "G";
    vocab[2].word = str;

    vocab[3].cn = 4;
    str = "R";
    vocab[3].word = str;

    vocab[4].cn = 3;
    str = "O";
    vocab[4].word = str;

    vocab[5].cn = 2;
    str = "F";
    vocab[5].word = str;


    CreateBinaryTree();

    for (int a = 0; a < vocab_size; a++) {
        printf("word=%s\t", vocab[a].word);
        printf("cn=%d\t", vocab[a].cn);
        printf("codelen=%d\t", vocab[a].codelen);
        
        printf("code=");
        for(int i = 0; i < vocab[a].codelen; i++) {
            printf("%d", vocab[a].code[i]);
        }
        printf("\t");

        printf("point=");
        for(int i = 0; i < vocab[a].codelen; i++) {
            printf("%d-", vocab[a].point[i]);
        }
        
        printf("\n");
         //printf("point=%s\n", vocab[a].point);

    }

    
    
	return 0;
}
