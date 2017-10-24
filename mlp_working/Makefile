CC=g++
CFLAGS=-g -Wall -std=c++11
LDFLAGS=-lconfig++
RM=rm -rf

SRCS=$(wildcard *.cc)
HDRS=$(wildcard *.h)
OBJS=$(SRCS:.cc=.o)
EXEC=mlp

default: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cc $(HDRS)
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	$(RM) $(OBJS) $(EXEC)

