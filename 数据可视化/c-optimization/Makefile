# Makefile for C Optimization Module
# 数据可视化系统 - C语言性能优化模块

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -mtune=native -fopenmp
LDFLAGS = -lm -fopenmp
TARGET = libdata_optimization.so
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# 源文件
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# 头文件
HEADERS = $(wildcard $(INCDIR)/*.h)

# 默认目标
all: $(BINDIR)/$(TARGET)

# 创建目录
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# 编译目标文件
$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) | $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# 链接生成共享库
$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) -shared $(OBJECTS) -o $@ $(LDFLAGS)

# 编译测试程序
test: $(BINDIR)/test_optimization
	./$(BINDIR)/test_optimization

$(BINDIR)/test_optimization: $(SRCDIR)/test_optimization.c $(HEADERS) | $(BINDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -o $@ $(LDFLAGS)

# 编译示例程序
example: $(BINDIR)/example
	./$(BINDIR)/example

$(BINDIR)/example: $(SRCDIR)/example.c $(HEADERS) | $(BINDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -o $@ $(LDFLAGS)

# 性能测试
benchmark: $(BINDIR)/benchmark
	./$(BINDIR)/benchmark

$(BINDIR)/benchmark: $(SRCDIR)/benchmark.c $(HEADERS) | $(BINDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -o $@ $(LDFLAGS)

# 清理
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# 安装
install: $(BINDIR)/$(TARGET)
	cp $(BINDIR)/$(TARGET) /usr/local/lib/
	ldconfig

# 卸载
uninstall:
	rm -f /usr/local/lib/$(TARGET)
	ldconfig

# 帮助
help:
	@echo "可用的目标:"
	@echo "  all        - 编译共享库 (默认)"
	@echo "  test       - 编译并运行测试程序"
	@echo "  example    - 编译并运行示例程序"
	@echo "  benchmark  - 编译并运行性能测试"
	@echo "  clean      - 清理编译文件"
	@echo "  install    - 安装共享库到系统"
	@echo "  uninstall  - 从系统卸载共享库"
	@echo "  help       - 显示此帮助信息"

# 声明伪目标
.PHONY: all test example benchmark clean install uninstall help
