target := crt.o syscalls.o

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%.o: %.S
	$(CC) $(CFLAGS) -c $<

.PHONY: all clean

all: $(target)

clean:
	rm -rf *.o