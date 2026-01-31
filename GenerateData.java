import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Fast Text Data Generator - Java Version
 * Generates comma-separated integers using all CPU cores.
 * 
 * Much faster than Python due to true multi-threading (no GIL).
 */
public class GenerateData {
    
    private static final int DEFAULT_WORKERS = Runtime.getRuntime().availableProcessors();
    
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            printUsage();
            System.exit(1);
        }
        
        String filepath = args[0];
        double sizeGB = 1.0;
        int numWorkers = DEFAULT_WORKERS;
        long knownMax = Long.MIN_VALUE;
        boolean hasKnownMax = false;
        String mode = "random";
        
        // Parse arguments
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--size":
                    if (i + 1 < args.length) sizeGB = Double.parseDouble(args[++i]);
                    break;
                case "--workers":
                    if (i + 1 < args.length) numWorkers = Integer.parseInt(args[++i]);
                    break;
                case "--known-max":
                    if (i + 1 < args.length) {
                        knownMax = Long.parseLong(args[++i]);
                        hasKnownMax = true;
                    }
                    break;
                case "--mode":
                    if (i + 1 < args.length) mode = args[++i];
                    break;
            }
        }
        
        if (numWorkers == 0) {
            numWorkers = DEFAULT_WORKERS;
        }
        
        generateTextFile(filepath, sizeGB, numWorkers, hasKnownMax ? knownMax : null, mode);
    }
    
    private static void printUsage() {
        System.out.println("Usage: java GenerateData <filepath> [options]");
        System.out.println("\nOptions:");
        System.out.println("  --size GB        Target file size in GB (default: 1)");
        System.out.println("  --workers N      Number of worker threads (default: all cores)");
        System.out.println("  --known-max N    Insert known maximum value");
        System.out.println("  --mode MODE      Generation mode: random, pattern (default: random)");
        System.out.println("\nExamples:");
        System.out.println("  java GenerateData data.txt --size 1");
        System.out.println("  java GenerateData data.txt --size 10 --workers 8");
        System.out.println("  java GenerateData data.txt --size 1 --known-max 999999999999999");
    }
    
    private static void generateTextFile(String filepath, double sizeGB, int numWorkers,
                                          Long knownMax, String mode) throws Exception {
        
        // Estimate number of integers needed
        // Average integer string length ~16 chars + comma = ~17 bytes
        long targetBytes = (long)(sizeGB * 1024 * 1024 * 1024);
        int avgBytesPerInt = 17;
        long totalIntegers = targetBytes / avgBytesPerInt;
        
        // Round to make even division
        totalIntegers = (totalIntegers / numWorkers) * numWorkers;
        long integersPerWorker = totalIntegers / numWorkers;
        
        System.out.println("\n" + "=".repeat(70));
        System.out.println("Text Data Generator (Java) - Mode: " + mode.toUpperCase());
        System.out.println("=".repeat(70));
        System.out.printf("Output: %s%n", filepath);
        System.out.printf("Target Size: ~%.2f GB%n", sizeGB);
        System.out.printf("Estimated Integers: %,d%n", totalIntegers);
        System.out.printf("Workers: %d%n", numWorkers);
        System.out.printf("Integers per Worker: %,d%n", integersPerWorker);
        if (knownMax != null) {
            System.out.printf("Known Max: %d%n", knownMax);
        }
        System.out.println("=".repeat(70) + "\n");
        
        // Determine which worker gets the known max
        int maxWorkerIndex = -1;
        long maxPositionInWorker = 0;
        if (knownMax != null) {
            Random rng = new Random(42);
            long globalPos = rng.nextLong(totalIntegers);
            maxWorkerIndex = (int)(globalPos / integersPerWorker);
            maxPositionInWorker = globalPos % integersPerWorker;
            System.out.printf("Known max will be in segment %d at position %d%n", 
                             maxWorkerIndex, maxPositionInWorker);
        }
        
        // Create temporary files for each worker
        System.out.println("Creating worker segments...");
        File[] tempFiles = new File[numWorkers];
        for (int i = 0; i < numWorkers; i++) {
            tempFiles[i] = File.createTempFile("segment_" + i + "_", ".tmp");
            tempFiles[i].deleteOnExit();
        }
        
        // Track progress
        AtomicLong totalBytesWritten = new AtomicLong(0);
        AtomicInteger completedWorkers = new AtomicInteger(0);
        
        long startTime = System.nanoTime();
        
        // Start progress monitor thread
        Thread progressThread = new Thread(() -> {
            long lastBytes = 0;
            long lastTime = System.nanoTime();
            
            while (!Thread.interrupted()) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
                
                long currentBytes = totalBytesWritten.get();
                long currentTime = System.nanoTime();
                int completed = completedWorkers.get();
                
                double deltaBytes = currentBytes - lastBytes;
                double deltaTime = (currentTime - lastTime) / 1e9;
                
                if (deltaTime > 0) {
                    double speedMBps = (deltaBytes / (1024.0 * 1024.0)) / deltaTime;
                    double totalGB = currentBytes / (1024.0 * 1024.0 * 1024.0);
                    double pct = (completed * 100.0) / numWorkers;
                    
                    System.out.printf("\r[%5.1f%%] %.2f GB | %.0f MB/s | %d/%d workers done",
                                     pct, totalGB, speedMBps, completed, numWorkers);
                    System.out.flush();
                }
                
                lastBytes = currentBytes;
                lastTime = currentTime;
            }
        });
        progressThread.setDaemon(true);
        progressThread.start();
        
        // Create worker tasks
        ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
        List<Future<Long>> futures = new ArrayList<>();
        
        final int fMaxWorkerIndex = maxWorkerIndex;
        final long fMaxPositionInWorker = maxPositionInWorker;
        final Long fKnownMax = knownMax;
        final String fMode = mode;
        
        for (int i = 0; i < numWorkers; i++) {
            final int workerId = i;
            final File tempFile = tempFiles[i];
            final long numInts = integersPerWorker;
            
            futures.add(executor.submit(() -> {
                long bytesWritten = 0;
                
                try (BufferedWriter writer = new BufferedWriter(
                        new FileWriter(tempFile), 8 * 1024 * 1024)) {  // 8MB buffer
                    
                    Random rng = new Random(42 + workerId * 7919);
                    StringBuilder batch = new StringBuilder(1024 * 1024);  // 1MB batch
                    int batchCount = 0;
                    int batchSize = 50000;  // Numbers per batch
                    
                    for (long j = 0; j < numInts; j++) {
                        long value;
                        
                        if (fMode.equals("pattern")) {
                            // Pattern mode: predictable sequence
                            value = (j + workerId * 100_000_000L) % 1_000_000_000_000_000L;
                        } else {
                            // Random mode
                            value = rng.nextLong(-1_000_000_000_000_000L, 1_000_000_000_000_000L);
                        }
                        
                        // Insert known max if this is the right position
                        if (fKnownMax != null && workerId == fMaxWorkerIndex && j == fMaxPositionInWorker) {
                            value = fKnownMax;
                        }
                        
                        // Add comma separator (except for first number of first worker)
                        if (j > 0 || workerId > 0) {
                            batch.append(',');
                        }
                        batch.append(value);
                        batchCount++;
                        
                        // Write batch to file
                        if (batchCount >= batchSize) {
                            String batchStr = batch.toString();
                            writer.write(batchStr);
                            bytesWritten += batchStr.length();
                            totalBytesWritten.addAndGet(batchStr.length());
                            batch.setLength(0);
                            batchCount = 0;
                        }
                    }
                    
                    // Write remaining batch
                    if (batch.length() > 0) {
                        String batchStr = batch.toString();
                        writer.write(batchStr);
                        bytesWritten += batchStr.length();
                        totalBytesWritten.addAndGet(batchStr.length());
                    }
                }
                
                completedWorkers.incrementAndGet();
                return bytesWritten;
            }));
        }
        
        // Wait for all workers to complete
        long totalBytes = 0;
        for (Future<Long> future : futures) {
            totalBytes += future.get();
        }
        
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);
        
        // Stop progress thread
        progressThread.interrupt();
        progressThread.join(1000);
        
        long generationTime = System.nanoTime() - startTime;
        
        System.out.println();
        System.out.println("\nMerging segments into final file...");
        
        // Merge all temp files into final file
        long mergeStart = System.nanoTime();
        try (BufferedOutputStream out = new BufferedOutputStream(
                new FileOutputStream(filepath), 8 * 1024 * 1024)) {
            
            byte[] buffer = new byte[8 * 1024 * 1024];  // 8MB buffer
            
            for (int i = 0; i < numWorkers; i++) {
                try (BufferedInputStream in = new BufferedInputStream(
                        new FileInputStream(tempFiles[i]), 8 * 1024 * 1024)) {
                    
                    int bytesRead;
                    while ((bytesRead = in.read(buffer)) != -1) {
                        out.write(buffer, 0, bytesRead);
                    }
                }
                
                // Delete temp file after merging
                tempFiles[i].delete();
                
                double pct = (i + 1) * 100.0 / numWorkers;
                System.out.printf("\rMerging: [%5.1f%%] %d/%d segments", pct, i + 1, numWorkers);
                System.out.flush();
            }
        }
        
        long mergeTime = System.nanoTime() - mergeStart;
        long totalTime = System.nanoTime() - startTime;
        
        // Get final file size
        File outputFile = new File(filepath);
        long actualSize = outputFile.length();
        
        double genTimeSec = generationTime / 1e9;
        double mergeTimeSec = mergeTime / 1e9;
        double totalTimeSec = totalTime / 1e9;
        double speedMBps = (actualSize / (1024.0 * 1024.0)) / totalTimeSec;
        
        System.out.println("\n");
        System.out.println("=".repeat(70));
        System.out.println("GENERATION COMPLETE");
        System.out.println("=".repeat(70));
        System.out.printf("File Size: %.2f GB (%,d bytes)%n", actualSize / (1024.0*1024.0*1024.0), actualSize);
        System.out.printf("Total Integers: %,d%n", totalIntegers);
        System.out.printf("Generation Time: %.1f seconds%n", genTimeSec);
        System.out.printf("Merge Time: %.1f seconds%n", mergeTimeSec);
        System.out.printf("Total Time: %.1f seconds%n", totalTimeSec);
        System.out.printf("Overall Speed: %.0f MB/s%n", speedMBps);
        if (knownMax != null) {
            System.out.printf("Known Max: %d%n", knownMax);
        }
        System.out.println("=".repeat(70) + "\n");
        
        System.out.printf("To find max: java MaxFinderText %s --threads 0%n", filepath);
        System.out.printf("         or: python max_finder_text.py %s --threads 0%n", filepath);
    }
}
