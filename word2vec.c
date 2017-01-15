//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.



//--------------------------------------------------
// comment by junGle. 2015.11.3
// download url: 
//      https://github.com/d0evi1/word2vec_insight
//--------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers


/**
 * word与Huffman树编码
 */
struct vocab_word {
  long long cn;     // 词在训练集中的词频率
  int *point;       // 编码的节点路径
  char *word,       // 词
       *code,       // Huffman编码，0、1串
       codelen;     // Huffman编码长度
};

char train_file[MAX_STRING], 
     output_file[MAX_STRING];

char save_vocab_file[MAX_STRING], 
     read_vocab_file[MAX_STRING];

struct vocab_word *vocab;

// 
int binary = 0, 
    cbow = 1, 
    debug_mode = 2, 
    window = 5, 
    min_count = 5, 
    num_threads = 12, 
    min_reduce = 1;     // min_reduce 

// 
int *vocab_hash;

// 
long long vocab_max_size = 1000,    // 
     vocab_size = 0,                // 输入层size：即词汇表size.
     layer1_size = 100;             // hidden层的size

//
long long train_words = 0, 
     word_count_actual = 0,
     iter = 5,                  // 缺省配置. 迭代次数.
     file_size = 0, 
     classes = 0;

// 学习率.
real alpha = 0.025,         // 学习率. 缺省:0.025 
     starting_alpha,        // 起始学习率. 等于初始设置的alpha，alpha在学习过程中会变化. 
     sample = 1e-3;         // subsample. 缺省：0.001

// 
real *syn0,         // input -> hidden 的 weights，大小：vocab_size * layer1_size 
     *syn1,         // hidden->output 的 weights，大小：vocab_size * layer1_size 
     *syn1neg,      // 
     *expTable;

// 
clock_t start;

// 默认配置.
int hs = 0, 
    negative = 5;

// 1-gram table.
const int table_size = 1e8;
int *table;


/**
 * unigram/1-gram: 每个单词的cn^pow表，负样本抽样中用到
 *
 * @return：table
 * 
 * table_size大小: 1亿 
 *  
 * magic num: power=0.75
 * cn^pow: 可将它看成是衰减词频，对词频做一定衰减，比词频稍微小一点
 *
 * 举个例子：
 *
 * 假如输入词频:    F:9     W:5     C:3     D:2
 * 则衰减词频为:    F:5.2   W:3.34  C:2.28  D:1.68      ∑cnt^0.75=12.5 
 * 相应的比例为:    F:0.42  W:0.27  C:0.18  D:0.13
 * 
 * 最终的table结果：
 *      'F'占比: 0.42, 最前段位置的42%格子，都是F
 *      'W'占比: 0.27, 次前段位置的27%格子，都是W
 *      'C'占比: 0.18, 第三段位置的18%格子，都是C
 *      'D'占比: 0.13, 第四段位置的13%格子，都是D
 */
void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;

  // 分配内存.
  table = (int *)malloc(table_size * sizeof(int));

  // power: train_words_pow = ∑(cn^0.75)
  // train_words_pow：衰减后的总词频数
  for (a = 0; a < vocab_size; a++) {
      train_words_pow += pow(vocab[a].cn, power);
  }

  // 第0个词的衰减词频(归一化).
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;

  // 遍历表的所有格子，添加相应的词汇表索引
  for (a = 0; a < table_size; a++) {

    // 持续填充某个索引，直到下面的条件打破
    table[a] = i;

    // 当该1-gram表中，格子所占空间 > 第i个的衰减词频占比
    // 则跳到下一个i，继续填充.
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    
    // 范围控制.
    if (i >= vocab_size) 
        i = vocab_size - 1;
  }
}

/**
 * 从文件中文件指针处，读取一个word.
 */
// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;

  //
  while (!feof(fin)) {
    // 读取一个字符.
    ch = fgetc(fin);

    // 回车键，继续
    if (ch == 13) continue;

    // 词边界：空格/tab/换行.
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      
      // 如果word有长度，则跳出循环，返回结果  
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      
      // 如果换行，且word为空，则为：</s>
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }

    // 词继续连在一起.最长的词：100.
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }


  // 结尾加上字符结尾符.
  word[a] = 0;
}

/*
 * 计算一个32位的hash值
 */
// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}


/** 
 * 搜索word对应在vocab中的索引.
 *
 */
// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {

  // 计算hash值.
  unsigned int hash = GetWordHash(word);

  // 检索对应的索引, 在vocab上的word，比较是否相等.
  // 如果找到，则返回对应的vocab索引.
  // 否则返回-1.
  while (1) {
    if (vocab_hash[hash] == -1) 
        return -1;
    
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) 
        return vocab_hash[hash];
    
    hash = (hash + 1) % vocab_hash_size;
  }

  return -1;
}

/**
 * 在文件指针fin上，读取一个word, 并返回它在vocab上的索引. 
 */
// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);

  if (feof(fin)) 
      return -1;
  
  return SearchVocab(word);
}

/**
 * 将一个word添加到词汇表中.
 */
// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  
  // vocab(word,cnt)  动态分配内存
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;

  // 如果vocab过小，重新分配内存.
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }

  // 计算该word的hash值. 保存对应word的词汇size.
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  
  // 返回当前词汇的size.
  return vocab_size - 1;
}

/**
 * qsort比较函数. 从大到小排序.
 */
// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

/**
 * 快速排序.
 */
// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;

  // 根据cnt进行排序 => vocab.(从大到小)
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  
  // 初始化.
  for (a = 0; a < vocab_hash_size; a++)
      vocab_hash[a] = -1;
  
  size = vocab_size;

  // 总训练words数.
  train_words = 0;
  
  // 比较min_count（判断是否抛弃），
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      
      // 冲突.
      while (vocab_hash[hash] != -1) 
          hash = (hash + 1) % vocab_hash_size;

      // 重置索引.
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }

  // 重新分配内存=>vocab.
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

  // 词汇表本身每个词汇，都是一个节点. node(code,point)
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

/*
 * 移除词汇表中，不频繁的token
 */
// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  
  // cnt < min_reduce: 释放内存.
  // 否则在内存上进行移位下.
  for (a = 0; a < vocab_size; a++) 
    if (vocab[a].cn > min_reduce) {
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    } else free(vocab[a].word);

  // 新的词汇size.
  vocab_size = b;

  // hash重置为-1.
  for (a = 0; a < vocab_hash_size; a++)
      vocab_hash[a] = -1;
  
  // 重新计算hash.
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);

    // 若冲突，hash+1.
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }

  // 
  fflush(stdout);
  min_reduce++;
}

/**
 * 使用词频创建一棵的Huffman树. 频率高的字将具有更短
 * 的Huffman二进制码(binary code).
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
  // binary: huffman二进制编码
  // parent_node: 
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

  // 初始化count数组的前一半
  for (a = 0; a < vocab_size; a++) 
      count[a] = vocab[a].cn;

  // 初始化count数组的后一半，用于交换. 赋很大值.
  for (a = vocab_size; a < vocab_size * 2; a++) 
      count[a] = 1e15;

  // 
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  
  // 最小值，pos1, pos2间进行比较.
  // [这里的设计有些精巧.]
  // 根据算法构建Huffman树，一次增加一个节点.
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {

    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {

      // 第1个：min1i. 遍历所有词汇的count，比较count；取较小值.
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

    // 第2个: min2i.
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

    // 为count/parent_node/binary赋值. 
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }

  // 将二进制编码分配给词汇表中每个词汇.
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;

    // 根据parent_node向上递归，获得Huffman编码：code. 
    // 以及路径:point
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;

      b = parent_node[b];
      if (b == vocab_size * 2 - 2) 
          break;
    }

    // 得到huffman编码长度
    vocab[a].codelen = i;

    // 得到word对应的Huffman编码code; 以及路径: point
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

/*
 * 从语料加生成词汇表.
 */
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;

  // 初始化词汇表.
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // 读取训练文件.
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  
  // 添加word到词汇表中.
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  
  // 循环读取文件 
  while (1) {

    // 读取一个词=> word
    ReadWord(word, fin);
    if (feof(fin)) break;

    // 训练word数，自增
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }

    // 搜索词汇表，返回索引，增加count数.
    i = SearchVocab(word);
    
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    
    // 如果词汇size过大，减小内存.
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  
  // sort排序
  SortVocab();

  // debug打印信息.
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }

  // 
  file_size = ftell(fin);
  fclose(fin);
}

/**
 * 保存词汇表到save_vocab_file中. (word, cnt)
 */
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

/**
 * 从read_vocab_file文件中读取词汇表.
 */
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];

  // 打开读取的词汇表.
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }

  // 读取word，存到内存vocab中
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }

  // 
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

/**
 * 神经网络
 * 参数：syn0, hs, negative 
 */
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;

  // a = vocab_size*layer_size*sizeof(real): 按128字节，数据对齐
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  
  // syn0
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  // hs: hierarchical softmax: syn1
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}

    // 初始化syn1, vocab_size x layer1_size, 权重为0
    for (a = 0; a < vocab_size; a++) 
        for (b = 0; b < layer1_size; b++) {
            syn1[a * layer1_size + b] = 0;
        }
  }

  // negative: syn1neg
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}

    // 初始化syn1neg, vocab_size x layer1_size, 权重为0
    for (a = 0; a < vocab_size; a++) 
        for (b = 0; b < layer1_size; b++) {
            syn1neg[a * layer1_size + b] = 0;
        }
  }

  // 初始化syn0矩阵, vocab_size x layer1_size, 随机分配权重
  // 权重大小范围：(-0.5/layer_size, 0.5/layer1_size) 
  for (a = 0; a < vocab_size; a++) 
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
      }

  // 创建Huffman二叉树.
  CreateBinaryTree();
}

/*
 * 训练模型线程.
 */
void *TrainModelThread(void *id) {
  long long a, 
       b, 
       d, 
       cw, 
       word, 
       last_word, 
       sentence_length = 0, 
       sentence_position = 0;

  long long word_count = 0, 
       last_word_count = 0, 
       sen[MAX_SENTENCE_LENGTH + 1];

  long long l1, 
       l2, 
       c, 
       target, 
       label, 
       local_iter = iter;

  // 随机数。将id做为起始值. 
  unsigned long long next_random = (long long)id;
  
  real f, 
       g;
  
  clock_t now;
  
  // step 1: 为neu1/neu1e分配内存.
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  
  // step 2: 打开训练文件. 定位到某线程id对应所属的文件段
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  // step 3: 训练主循环：
  // 每次读取1000个词到sen[]中，进行训练.
  //
  while (1) {

    // step 3-1: 更新要处理的word_count, last_word_count.
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;

      // a.打印调试信息: 学习率alpha, 进度, 每秒钟每个线程处理words数.
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      
      // b.自适应学习率.
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) {
          alpha = starting_alpha * 0.0001;
      }
    }
    
    // step 3-2: 从文件中读取1000个词，组成一个sentence.
    if (sentence_length == 0) {
      while (1) {

        // a.从文件中读取当前位置的词, 返回在vocab中的索引.  
        word = ReadWordIndex(fi);

        // b.文件末尾，结束
        if (feof(fi)) break;

        // c.索引不存在，抛弃该词，继续
        if (word == -1) continue;

        // 自增
        word_count++;

        // 为0，则结束
        if (word == 0) break;

        // d.进行subsampling，随机丢弃常见词，保持相同的频率排序ranking.
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {

          // 计算相应的抛弃概率ran.
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;

          // 生成一个随机数next_random.
          next_random = next_random * (unsigned long long)25214903917 + 11;

          // 如果random/65536 - ran > 0, 则抛弃该词，继续
          if (ran < (next_random & 0xFFFF) / (real)65536) 
              continue;
        }

        // e.将该词添加到句子sen中.最大长度1000.
        sen[sentence_length] = word;
        sentence_length++;

        if (sentence_length >= MAX_SENTENCE_LENGTH) 
            break;
      }
      
      sentence_position = 0;
    }

    // step 3-3: 如果到达文件末尾，或者word_count超过每个线程的train_words数，重新定位文件指针.
    if (feof(fi) || (word_count > train_words / num_threads)) {

      // a.更新 word_count_actual 
      word_count_actual += word_count - last_word_count;
      
      // b.更新 local_iter.
      local_iter--;
      if (local_iter == 0) 
          break;

      // c.重置0.
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;

      // d.重置文件指针，进行下一轮迭代.
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    // step 3-4: 获得句首词
    word = sen[sentence_position];
    if (word == -1) 
        continue;

    // step 3-5: 初始化neu1(隐层参数)、neu1e(隐层误差值).
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

    // step 3-6: 取window窗口的随机一个值.
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;

    // step 3-7: cbow模型.
    if (cbow) {  //train the cbow architecture
      // cbow模型:  正向传播，从输入层到隐层
      // in -> hidden
      
      
      // a: 通过cbow模型，使用当前窗口的上下文word，更新neu1
      //
      // 窗口大小随机：总大小为(window*2+1-2b); 左右两侧，各为window-b
      // b值范围：[0,window) 间随机.
      // cw: 滑动窗口中滑动过的有效词数.
      // 
      // 计算过程：
      //    一个窗口内，每个有效词，都会进行前向传播更新syn1、反向传播syn0.
      cw = 0;  
      for (a = b; a < window * 2 + 1 - b; a++) {
          if (a != window) {

            // a.1: 当前的word，刚好是窗口中心。从window-b开始，从左到右，且不能刚好是word自己  
            c = sentence_position - window + a;
            
            // a.2: c必须大于零，且必须小于sentence_length.
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            
            // a.3: 要处理的上下文word.
            last_word = sen[c];
            if (last_word == -1) continue;

            // a.4: 根据该窗口中某个上下文word(索引)，更新neu1.
            //      每个词输入，都与neu1上隐节点有连接
            //
            //      neu1 = ∑ 1*syn0     (neu1初始为0，不断更新; syn0初始随机)
            //
            for (c = 0; c < layer1_size; c++) 
                neu1[c] += syn0[c + last_word * layer1_size];

            // 自增cw.
            cw++;
          }
      }

      // b: 层次化softmax/negative sampling
      if (cw) {

        // 使用除以窗口长度. neu1 = ∑ syn0/cw  
        for (c = 0; c < layer1_size; c++) 
            neu1[c] /= cw;

        // b.1: 是否使用hs.（Hierarchical Softmax）
        if (hs) {

            // a.当前word的huffman编码长度, 沿着huffman编码路径向下行走.
            //      一次计算当前一个的中间节点
            for (d = 0; d < vocab[word].codelen; d++) {
              f = 0;

              // a.1: l2: 当前节点号 * 隐单元数，用于索引syn1权重.
              // 
              // 关于l2的最大值：
              //    假设该二叉树的每个中间节点，都有一个叶子节点
              //    此时它的最大中间数目为(vocab_size-1). 
              //    也就是说，huffman的最长编码为: (vocab_size-1)
              //
              // 因此，我们可知syn1的size: vocab_size x layer1_size
              // 是很合理的.
              // 每个节点，都对应着关于layer1_size的向量
              l2 = vocab[word].point[d] * layer1_size;
              
              // a.2: 前向传播: 计算f值，即前元隐单元和参数sync1的sum.
              //        1.(syn1刚开始全0) 
              //        2.f = ∑ neu1*sync1
              //
              // Propagate hidden -> output
              for (c = 0; c < layer1_size; c++) {
                  f += neu1[c] * syn1[c + l2];
              }

              // a.3: 再由f值作logistic regression计算，得到一个概率值f
              //    input:f (-6, 6) => logistic unit => output: f (0,1)
              //
              //    将 expTable的定义，与下式同时代入，可化简得到：
              //        expTable: 等于1/(1+e^-x)，即logistic函数
              //
              //    根据f=∑wx, 查expTable表(预先计算好的整数位e指数)，相当于做logistic运算.
              //    输入范围: (-MAX_EXP, MAX_EXP), 即(-6, 6)之间. 
              if (f <= -MAX_EXP) 
                  continue;
              else if (f >= MAX_EXP) 
                  continue;
              else 
                  f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

              // a.4: lr预测的f值(为1的概率), 与真实编码的反面的差. 即梯度 
              //    和该节点上真实编码位比较
              //    该公式可以由推导得到.
              // 'g' is the gradient multiplied by the learning rate
              g = (1 - vocab[word].code[d] - f) * alpha;
             
              // a.5: 反向传播: 利用当前节点所计算g和syn1，更新对应的:neu1e
              //    neu1e = ∑ g*syn1        (neu1e初始值全0, 不断更新)
              //
              // Propagate errors output -> hidden
              for (c = 0; c < layer1_size; c++) {
                  neu1e[c] += g * syn1[c + l2];
              }
              
              // a.6: 利用当前节点所计算的g和neu1, 更新syn1
              //    syn1 = ∑ g*neu1         (syn1初始值全0,不断更新)
              // Learn weights hidden -> output
              for (c = 0; c < layer1_size; c++) {
                  syn1[c + l2] += g * neu1[c];
              }
            }
        }

        // b.2: NEGATIVE SAMPLING,
        // negative为采样的数目.
        if (negative > 0) { 

            // 1个label=1的样本：       (word->word自身)
            // negative个label=0的样本: (word->非word本身的其它词)
            //
            // 原理：使用logistic regression来实现softmax
            for (d = 0; d < negative + 1; d++) {
              
              // 1.
              // 如果d=0, 初始化target/label
              // 如果d>0, 从词表中随机取一个词(非原词). label=0
              if (d == 0) {
                target = word;
                label = 1;
              } else {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                
                if (target == 0) 
                    target = next_random % (vocab_size - 1) + 1;
                
                if (target == word) 
                    continue;
                
                label = 0;
              }

              // 该target对应的位置
              l2 = target * layer1_size;
              f = 0;
              
              // 2. 目标：由词word，预测成target的概率
              // f = ∑ neu1 * syn1neg
              for (c = 0; c < layer1_size; c++) 
                  f += neu1[c] * syn1neg[c + l2];

              // 3. 计算logistic的概率, 并使用该g进行更新
              if (f > MAX_EXP) 
                  g = (label - 1) * alpha;
              else if (f < -MAX_EXP) 
                  g = (label - 0) * alpha;
              else 
                  g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

              // 4.使用g 更新neu1e
              for (c = 0; c < layer1_size; c++) 
                  neu1e[c] += g * syn1neg[c + l2];
              
              // 5.使用g 更新syn1neg
              for (c = 0; c < layer1_size; c++) 
                  syn1neg[c + l2] += g * neu1[c];
            }
        }

        // b.3: 反向传播, 根据当前窗口的上下文，利用neu1e，更新syn0
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) {
          if (a != window) {
              c = sentence_position - window + a;
              
              // a.范围判断
              if (c < 0) continue;
              if (c >= sentence_length) continue;

              // b.获取窗口当前词对应索引，用于检索对应的syn0.
              last_word = sen[c];
              if (last_word == -1) continue;

              // c.更新窗口当前词所影响的syn0权重.
              for (c = 0; c < layer1_size; c++) 
                  syn0[c + last_word * layer1_size] += neu1e[c];
              
          }
        }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    
    // step 3-6: 移动当前句子(1000个词)中的当前词汇指针: 移动一格.
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }

  // 
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

/*
 * 训练模型.
 */
void TrainModel() {
  long a, b, c, d;
  FILE *fo;

  // a. 使用多少线程.
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  
  // b. 学习率: default: skip-gram=0.025; CBOW = 0.05
  starting_alpha = alpha;

  // b. 如果设置了词汇表，使用自定义词汇表；
  //    否则，使用语料库中生成的词汇表； 
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  
  // c. 是否保存词汇表
  if (save_vocab_file[0] != 0) SaveVocab();

  // 必须设置输出文件.
  if (output_file[0] == 0) return;

  // d. 初始化神经网络参数.
  InitNet();

  // e.初始化unigram表.
  if (negative > 0) InitUnigramTable();

  start = clock();

  // f. 多线程训练：读取整个文件，进行神经网络模型训练.
  for (a = 0; a < num_threads; a++) 
      pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) 
      pthread_join(pt[a], NULL);
  
  // g. 结果输出.
  fo = fopen(output_file, "wb");

  // g1: 不使用分类，保存词向量embedding.
  if (classes == 0) {
    // g1.1: 保存头一行
    //      vocab_size:  词汇size
    //      layer1_size: 隐层size
    // Save the word vectors: 
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    
    // g1.2: 保存每一词汇，对应的syn0的参数: 这些syn0参数即构成词向量
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) {
          // 普通c二进制保存
          for (b = 0; b < layer1_size; b++) {
              fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
          }
      }
      else {
          // 文本方式保存
          for (b = 0; b < layer1_size; b++) {
              fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
          }
      }
      fprintf(fo, "\n");
    }
  } 
  else {  // g2: 使用分类，则保存

    // g2.1: 使用k-means在词向量上进行聚类
    // Run K-means on the word vectors
    int clcn = classes,     // 类别数 
        iter = 10,          // 迭代次数
        closeid;            // 中间变量

    // a. centcn: size=classes，表示每个类别上对应所属的词汇数
    int *centcn = (int *)malloc(classes * sizeof(int));
    
    // cl: 每个词汇量对应的类别
    int *cl = (int *)calloc(vocab_size, sizeof(int));
   
    // 中间变量 
    real closev, x;     

    // cent: classes * layer1_size, 即：类别数 x 向量维度
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

    // b.初始划分各个簇：遍历所有词汇，将它们分配按模分配
    for (a = 0; a < vocab_size; a++) 
        cl[a] = a % clcn;
    
    // c.进行k-means迭代
    for (a = 0; a < iter; a++) {
      
      // c.1: 重置cent.
      for (b = 0; b < clcn * layer1_size; b++) 
          cent[b] = 0;
      
      // c.2: 重置centcn.
      for (b = 0; b < clcn; b++) 
          centcn[b] = 1;

      // c.3: 遍历每个词汇, 作两个事情：
      //    1.在该词所在分类上，叠加上该词所对应词向量各维度分量.
      //    2.累积每个类别上的词数.
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
            cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        }

        centcn[cl[c]]++;
      }

      // c.4: 遍历每个类别，对词向量空间维度上求得：归一化cent向量和.
      for (b = 0; b < clcn; b++) {
        closev = 0;

        // 1.
        // cent:将该类别下的向量各维度上的总量做平均(除以对应类别上的词数)
        // closev: 等于平均后的平方和
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }

        // 2.做sqrt根号运算.
        closev = sqrt(closev);

        // 3.再除以该量，进行归一化.
        for (c = 0; c < layer1_size; c++) 
            cent[layer1_size * b + c] /= closev;
      }

      // c.5: 遍历所有词汇.
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        
        // 遍历每个分类.
        for (d = 0; d < clcn; d++) {
          x = 0;

          // 点积运算：
          // x = ∑ 该类别对应的归一化cent向量和 * 该词对应向量分量syn0
          // 
          // 两者都已经归一化
          //
          // >0, 表示：两个向量同向
          // <0, 表示：两个向量反向
          for (b = 0; b < layer1_size; b++) {
              x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          }
          
          // 如果x > closev, 慢慢纠正，则更新closev=x
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }

        // c.6: 该词汇所对应的分类=closeid，进入下一轮迭代.
        cl[c] = closeid;
      }
    }

    // d.保存k-means结果，保存(词,分类).
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) 
        fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);

    
    free(centcn);
    free(cent);
    free(cl);
  }

  // close文件
  fclose(fo);
}

/*
 * 解析命令行参数. ./word2vec -file a.txt ... => 
 */
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) {
      if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
      }
  }
  
  return -1;
}

/*
 * 主函数.
 */
int main(int argc, char **argv) {
  int i;

  // step 1:  
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }

  // step 2: 读取命令行参数. 
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  
  // step 3: 分配空间.
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

  // step 4: 分配logistic查表.
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  
  // 初始化: 预先计算好指数运算表. 
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }

  // step 5: 训练模型.
  TrainModel();
  return 0;
}
