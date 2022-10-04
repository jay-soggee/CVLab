#include "sep_30_LL_corner.h"
#include <stdio.h>
#include <stdlib.h>

extern node* TAIL;

node* initList(node* tail)
{
	node* head = (node*)calloc(1, sizeof(node));
	head->next = tail;
	tail->next = tail;
	return head;
}

/* returns pointer of it's histogram */
node* insertAfter(node* t, int x, int y, double r)
{
	node* temp;
	temp = (node*)calloc(1, sizeof(node));
	temp->x = x;
	temp->y = y;
	temp->R = r;
	temp->hist = (double*)calloc(36, sizeof(node));
	temp->next = t->next;
	t->next = temp;
	return temp;
}

/* usage : while(deleteAfter(node)); */
int deleteAfter(node* t)
{
	node* temp;
	if (t->next == TAIL)
		return 0;
	temp = t->next;
	free(temp->hist);
	t->next = t->next->next;
	free(temp);
	return 1;
}