#ifndef LLCORNER
#define LLCORNER

typedef struct _node {
	int x;
	int y;
	double R;
	double* hist;
	struct _node* sim_to;
	struct _node* next;
}node;


node* initList(node* tail);

node* insertAfter(node* t, int x, int y, double r);

int deleteAfter(node* t);



#endif
