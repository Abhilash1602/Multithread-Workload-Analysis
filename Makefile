CC = gcc
CFLAGS = -Wall -Werror

# List all C files and derive executable names
SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=)

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) $< -o $@

run: all
	@for exe in $(TARGETS); do \
		echo "Running $$exe:"; \
		./$$exe; \
	done

clean:
	rm -f $(TARGETS)
