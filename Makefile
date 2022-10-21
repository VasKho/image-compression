CC := gcc
CFLAGS = -O2
LSO := -lpng -lm

BUILD_DIR := .
SRC_DIRS := ./src

OUT_BIN := ${BUILD_DIR}/main

SRCS := $(shell find $(SRC_DIRS) -name '*.c')

install: main.c ${SRCS}
	${CC} main.c ${SRCS} ${CFLAGS} ${LSO} -o ${OUT_BIN}

debug: main.c ${SRCS}
	${CC} main.c ${SRCS} ${CFLAGS} ${LSO} -g3 -o ${OUT_BIN}

test:
	$(shell valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ${OUT_BIN} ./test.png)

clean:
	rm ${OUT_BIN}
