package org.example;

import java.nio.file.*;

// 这是在B2上的修改2
class GraphExporter {
    // 需要系统安装Graphviz并配置PATH环境变量
    public static void exportToImage(Graph graph, String format) throws Exception {
        String dotContent = graph.toDOT();
        Path dotFile = Files.createTempFile("graph", ".dot");
        Files.write(dotFile, dotContent.getBytes());

        String outputFile = "./src/test/graph." + format;
        Process process = new ProcessBuilder("dot", "-T" + format,
                dotFile.toString(), "-o", outputFile).start();
        int exitCode = process.waitFor();

        if (exitCode == 0) {
            System.out.println("图形文件已生成: " + outputFile);
        } else {
            System.err.println("生成失败，请检查是否安装Graphviz");
        }
    }
}

