import java.io.*;
import java.lang.management.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

/**
 * Text File Maximum Number Finder - Java Version
 * Finds the maximum integer in a comma-separated text file.
 * 
 * Supports:
 * - Single-threaded and multi-threaded modes
 * - Real-time throughput monitoring
 * - Per-thread CPU time tracking
 */
public class MaxFinderText {
    
    private static final AtomicLong bytesRead = new AtomicLong(0);
    private static final AtomicLong numbersRead = new AtomicLong(0);
    private static volatile boolean monitoringActive = false;
    
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            printUsage();
            System.exit(1);
        }
        
        String filepath = args[0];
        int numThreads = 1;
        int chunkSizeMB = 64;
        String csvPath = null;
        
        // Parse arguments
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--threads":
                    if (i + 1 < args.length) numThreads = Integer.parseInt(args[++i]);
                    break;
                case "--chunk-size":
                    if (i + 1 < args.length) chunkSizeMB = Integer.parseInt(args[++i]);
                    break;
                case "--csv":
                    if (i + 1 < args.length) csvPath = args[++i];
                    break;
            }
        }
        
        if (numThreads == 0) {
            numThreads = Runtime.getRuntime().availableProcessors();
            System.out.println("Using all " + numThreads + " logical cores");
        }
        
        File file = new File(filepath);
        if (!file.exists()) {
            System.err.println("Error: File not found: " + filepath);
            System.exit(1);
        }
        
        runBenchmark(filepath, numThreads, chunkSizeMB, csvPath);
    }
    
    private static void printUsage() {
        System.out.println("Usage: java MaxFinderText <filepath> [options]");
        System.out.println("\nOptions:");
        System.out.println("  --threads N      Number of threads (default: 1, 0 = all cores)");
        System.out.println("  --chunk-size MB  Read chunk size in MB (default: 64)");
        System.out.println("  --csv FILE       Append results to CSV file");
        System.out.println("\nExamples:");
        System.out.println("  java MaxFinderText data.txt --threads 1");
        System.out.println("  java MaxFinderText data.txt --threads 10");
        System.out.println("  java MaxFinderText data.txt --threads 0  # all cores");
    }
    
    private static void runBenchmark(String filepath, int numThreads, int chunkSizeMB,
                                      String csvPath) throws Exception {
        
        File file = new File(filepath);
        long fileSize = file.length();
        int chunkSize = chunkSizeMB * 1024 * 1024;
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Text File Max Finder Benchmark (Java)");
        System.out.println("=".repeat(60));
        System.out.printf("File: %s%n", filepath);
        System.out.printf("File Size: %.2f GB%n", fileSize / (1024.0 * 1024.0 * 1024.0));
        System.out.printf("Mode: %s%n", numThreads == 1 ? "Single-threaded" : "Multi-threaded (" + numThreads + " threads)");
        System.out.printf("Format: Comma-separated integers (text)%n");
        System.out.printf("Available Processors: %d%n", availableProcessors);
        System.out.println("=".repeat(60) + "\n");
        
        // Reset counters
        bytesRead.set(0);
        numbersRead.set(0);
        monitoringActive = true;
        
        // Get thread MXBean for CPU time tracking
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        
        // Throughput samples
        List<Double> throughputSamples = Collections.synchronizedList(new ArrayList<>());
        
        // Start monitor thread
        Thread monitorThread = new Thread(() -> {
            long lastBytes = 0;
            long lastTime = System.nanoTime();
            
            while (monitoringActive) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    break;
                }
                
                long currentBytes = bytesRead.get();
                long currentNumbers = numbersRead.get();
                long currentTime = System.nanoTime();
                
                double deltaBytes = currentBytes - lastBytes;
                double deltaTime = (currentTime - lastTime) / 1e9;
                
                if (deltaTime > 0 && deltaBytes > 0) {
                    double speedMBps = (deltaBytes / (1024.0 * 1024.0)) / deltaTime;
                    throughputSamples.add(speedMBps);
                    
                    double totalGB = currentBytes / (1024.0 * 1024.0 * 1024.0);
                    System.out.printf("\r[Progress] Read: %.2f GB | Numbers: %,d | Throughput: %.0f MB/s",
                                     totalGB, currentNumbers, speedMBps);
                    System.out.flush();
                }
                
                lastBytes = currentBytes;
                lastTime = currentTime;
            }
        });
        monitorThread.setDaemon(true);
        monitorThread.start();
        
        // Record start times
        long wallStartTime = System.nanoTime();
        long cpuStartTime = threadMXBean.getCurrentThreadCpuTime();
        
        // Run the max finding
        long maxValue;
        Map<Long, Long> threadCpuTimes = new ConcurrentHashMap<>();
        
        if (numThreads == 1) {
            maxValue = findMaxSingleThread(filepath, chunkSize);
        } else {
            maxValue = findMaxMultiThread(filepath, numThreads, chunkSize, threadMXBean, threadCpuTimes);
        }
        
        // Record end times
        long wallEndTime = System.nanoTime();
        long cpuEndTime = threadMXBean.getCurrentThreadCpuTime();
        
        // Stop monitoring
        monitoringActive = false;
        monitorThread.interrupt();
        monitorThread.join(1000);
        
        // Calculate times
        double wallTime = (wallEndTime - wallStartTime) / 1e9;
        double cpuTime = (cpuEndTime - cpuStartTime) / 1e9;
        
        // For multi-threaded, sum thread CPU times
        if (!threadCpuTimes.isEmpty()) {
            double totalThreadCpu = 0;
            for (Long t : threadCpuTimes.values()) {
                totalThreadCpu += t / 1e9;
            }
            cpuTime = totalThreadCpu;
        }
        
        double ioWaitTime = Math.max(0, wallTime - cpuTime);
        double ioWaitPct = (ioWaitTime / wallTime) * 100;
        
        // Calculate throughput stats
        double avgThroughput = 0, peakThroughput = 0;
        if (!throughputSamples.isEmpty()) {
            for (double t : throughputSamples) {
                avgThroughput += t;
                peakThroughput = Math.max(peakThroughput, t);
            }
            avgThroughput /= throughputSamples.size();
        }
        double overallThroughput = (bytesRead.get() / (1024.0 * 1024.0)) / wallTime;
        
        // Print results
        System.out.println("\n\n" + "=".repeat(80));
        System.out.printf("%40s%n", "BENCHMARK RESULTS SUMMARY");
        System.out.println("=".repeat(80));
        
        System.out.printf("%n%-40s %35s%n", "METRIC", "VALUE");
        System.out.printf("%-40s %35s%n", "-".repeat(40), "-".repeat(35));
        
        System.out.printf("%-40s %32.2f GB%n", "File Size", fileSize / (1024.0 * 1024.0 * 1024.0));
        System.out.printf("%-40s %35d%n", "Threads Used", numThreads);
        System.out.printf("%-40s %35d%n", "Maximum Value Found", maxValue);
        System.out.printf("%-40s %,35d%n", "Numbers Processed", numbersRead.get());
        System.out.printf("%-40s %32.2f sec%n", "Total Execution Time", wallTime);
        
        System.out.printf("%n%-40s%n", "--- DISK THROUGHPUT ---");
        System.out.printf("%-40s %30.2f MB/s%n", "Average Throughput", avgThroughput > 0 ? avgThroughput : overallThroughput);
        System.out.printf("%-40s %30.2f MB/s%n", "Peak Throughput", peakThroughput);
        System.out.printf("%-40s %30.2f MB/s%n", "Overall Throughput", overallThroughput);
        
        System.out.printf("%n%-40s%n", "--- I/O TIMING ---");
        System.out.printf("%-40s %32.2f sec%n", "Wall Clock Time", wallTime);
        System.out.printf("%-40s %32.2f sec%n", "CPU Time", cpuTime);
        System.out.printf("%-40s %32.2f sec%n", "I/O Wait Time (estimated)", ioWaitTime);
        System.out.printf("%-40s %32.1f %%%n", "I/O Wait Percentage", ioWaitPct);
        
        System.out.printf("%n%-40s%n", "--- SYSTEM INFO ---");
        System.out.printf("%-40s %35d%n", "Available Processors", availableProcessors);
        System.out.printf("%-40s %35d%n", "Threads Used", numThreads);
        
        System.out.println("\n" + "=".repeat(80) + "\n");
        
        System.out.printf("Summary: Found max=%d in %.2fs @ %.0f MB/s avg throughput%n",
                         maxValue, wallTime, overallThroughput);
        
        // Save to CSV if requested
        if (csvPath != null) {
            saveResultsCsv(csvPath, fileSize, numThreads, maxValue, wallTime, 
                          avgThroughput > 0 ? avgThroughput : overallThroughput,
                          cpuTime, ioWaitTime, ioWaitPct);
            System.out.println("Results appended to: " + csvPath);
        }
    }
    
    private static long findMaxSingleThread(String filepath, int chunkSize) throws Exception {
        long maxVal = Long.MIN_VALUE;
        
        // Use smaller buffer to avoid memory issues
        int bufferSize = Math.min(chunkSize, 8 * 1024 * 1024);  // Max 8MB
        
        try (BufferedInputStream bis = new BufferedInputStream(
                new FileInputStream(filepath), bufferSize)) {
            
            StringBuilder currentNumber = new StringBuilder(32);
            byte[] byteBuffer = new byte[bufferSize];
            int bytesReadCount;
            
            while ((bytesReadCount = bis.read(byteBuffer)) != -1) {
                bytesRead.addAndGet(bytesReadCount);
                
                // Process byte by byte
                for (int i = 0; i < bytesReadCount; i++) {
                    byte b = byteBuffer[i];
                    
                    if (b == ',') {
                        // End of number
                        if (currentNumber.length() > 0) {
                            try {
                                long val = Long.parseLong(currentNumber.toString().trim());
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                                numbersRead.incrementAndGet();
                            } catch (NumberFormatException e) {
                                // Skip invalid
                            }
                            currentNumber.setLength(0);
                        }
                    } else {
                        currentNumber.append((char) b);
                    }
                }
            }
            
            // Process last number
            if (currentNumber.length() > 0) {
                try {
                    long val = Long.parseLong(currentNumber.toString().trim());
                    if (val > maxVal) {
                        maxVal = val;
                    }
                    numbersRead.incrementAndGet();
                } catch (NumberFormatException e) {
                    // Skip invalid
                }
            }
        }
        
        return maxVal;
    }
    
    private static long findMaxMultiThread(String filepath, int numThreads, int chunkSize,
                                            ThreadMXBean threadMXBean,
                                            Map<Long, Long> threadCpuTimes) throws Exception {
        
        File file = new File(filepath);
        long fileSize = file.length();
        long segmentSize = fileSize / numThreads;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Long>> futures = new ArrayList<>();
        
        for (int i = 0; i < numThreads; i++) {
            final long start = i * segmentSize;
            final long end = (i == numThreads - 1) ? fileSize : (i + 1) * segmentSize;
            
            futures.add(executor.submit(() -> {
                long threadId = Thread.currentThread().getId();
                long cpuStart = threadMXBean.getThreadCpuTime(threadId);
                
                long result = findMaxInRange(filepath, start, end, chunkSize);
                
                long cpuEnd = threadMXBean.getThreadCpuTime(threadId);
                threadCpuTimes.put(threadId, cpuEnd - cpuStart);
                
                return result;
            }));
        }
        
        long maxVal = Long.MIN_VALUE;
        for (Future<Long> future : futures) {
            long result = future.get();
            if (result > maxVal) {
                maxVal = result;
            }
        }
        
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);
        
        return maxVal;
    }
    
    private static long findMaxInRange(String filepath, long start, long end, int chunkSize) throws Exception {
        long maxVal = Long.MIN_VALUE;
        
        // Use smaller buffer to avoid memory issues
        int bufferSize = Math.min(chunkSize, 8 * 1024 * 1024);  // Max 8MB
        
        try (RandomAccessFile raf = new RandomAccessFile(filepath, "r")) {
            raf.seek(start);
            
            // Skip to next comma if not at start
            if (start > 0) {
                while (raf.getFilePointer() < end) {
                    int b = raf.read();
                    if (b == -1 || b == ',') break;
                }
            }
            
            StringBuilder currentNumber = new StringBuilder(32);  // For building current number
            byte[] byteBuffer = new byte[bufferSize];
            
            while (raf.getFilePointer() < end) {
                long remaining = end - raf.getFilePointer();
                int toRead = (int) Math.min(bufferSize, remaining);
                int bytesReadCount = raf.read(byteBuffer, 0, toRead);
                
                if (bytesReadCount <= 0) break;
                
                bytesRead.addAndGet(bytesReadCount);
                
                // Process byte by byte to avoid creating large strings
                for (int i = 0; i < bytesReadCount; i++) {
                    byte b = byteBuffer[i];
                    
                    if (b == ',') {
                        // End of number, parse it
                        if (currentNumber.length() > 0) {
                            try {
                                long val = Long.parseLong(currentNumber.toString().trim());
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                                numbersRead.incrementAndGet();
                            } catch (NumberFormatException e) {
                                // Skip invalid
                            }
                            currentNumber.setLength(0);
                        }
                    } else {
                        currentNumber.append((char) b);
                    }
                }
            }
            
            // Process last number
            if (currentNumber.length() > 0) {
                try {
                    long val = Long.parseLong(currentNumber.toString().trim());
                    if (val > maxVal) {
                        maxVal = val;
                    }
                    numbersRead.incrementAndGet();
                } catch (NumberFormatException e) {
                    // Skip invalid
                }
            }
        }
        
        return maxVal;
    }
    
    private static void saveResultsCsv(String csvPath, long fileSize, int numThreads,
                                        long maxValue, double wallTime, double throughput,
                                        double cpuTime, double ioWaitTime, double ioWaitPct) throws Exception {
        
        boolean fileExists = new File(csvPath).exists();
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(csvPath, true))) {
            if (!fileExists) {
                writer.println("timestamp,file_size_gb,num_threads,max_value,execution_time_sec," +
                              "avg_throughput_mbps,cpu_time_sec,io_wait_time_sec,io_wait_pct");
            }
            
            writer.printf("%s,%.4f,%d,%d,%.4f,%.2f,%.4f,%.4f,%.2f%n",
                         java.time.LocalDateTime.now().toString(),
                         fileSize / (1024.0 * 1024.0 * 1024.0),
                         numThreads,
                         maxValue,
                         wallTime,
                         throughput,
                         cpuTime,
                         ioWaitTime,
                         ioWaitPct);
        }
    }
}
