# aws-s3-batch-uploader 🚀

A robust, high-performance Python tool for uploading massive amounts of files to Amazon S3 with confidence.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## The Problem This Solves

Let's face it – uploading hundreds of thousands of files to S3 is a special kind of headache. You start a simple script,
go grab coffee, return to find it crashed halfway through, and now you're playing the "which files actually made it?"
detective game. Not anymore!

This tool was born from the battle scars of managing 50k+ directories containing 400k+ images. It handles the
complexities so you don't have to.

## ✨ Key Features

- **Parallel Processing**: Concurrent uploads to maximize throughput
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Structure Preservation**: Maintains your folder hierarchy in S3
- **Comprehensive Logging**: Detailed logs of everything that happens
- **Performance Metrics**: Track speed, progress, and completion statistics
- **Selective Uploads**: Upload only specific files from a provided list
- **Dry Run Mode**: Test your setup without actually uploading anything

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/poacosta/aws-s3-batch-uploader.git
cd aws-s3-batch-uploader

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- boto3
- tqdm

## 🚀 Quick Start

1. Ensure your AWS credentials are configured (via environment variables, ~/.aws/credentials, or IAM role)
2. Create a text file listing paths to upload (one path per line)
3. Run the uploader:

```bash
python s3_uploader.py paths_to_upload.txt my-bucket-name
```

That's it! The tool will handle the rest, including logging, progress display, and error handling.

## 📋 Usage Examples

### Basic Upload

```bash
python s3_uploader.py file_list.txt my-bucket
```

### Preserve Directory Structure

```bash
python s3_uploader.py file_list.txt my-bucket --base-path /data/images
```

### Upload to a Specific Folder in S3

```bash
python s3_uploader.py file_list.txt my-bucket --destination-prefix customer_uploads/batch1
```

### Maximum Performance

```bash
python s3_uploader.py file_list.txt my-bucket --max-concurrency 20
```

### Thorough Testing (Dry Run)

```bash
python s3_uploader.py file_list.txt my-bucket --dry-run --verbose
```

### CSV File with Headers

```bash
python s3_uploader.py image_paths.csv my-bucket --skip-header --path-column 2
```

## 🎮 Command-Line Options

| Argument               | Description                                     | Default         |
|------------------------|-------------------------------------------------|-----------------|
| `file_list`            | Path to CSV or TXT file with files to upload    | (Required)      |
| `bucket`               | S3 bucket name                                  | (Required)      |
| `--base-path`          | Base path for calculating relative paths        | None            |
| `--destination-prefix` | Prefix for all S3 object keys                   | None            |
| `--profile`            | AWS profile name to use                         | Default profile |
| `--region`             | AWS region                                      | us-east-1       |
| `--storage-class`      | S3 storage class                                | STANDARD        |
| `--path-column`        | Column index in CSV with file paths             | 0               |
| `--skip-header`        | Skip CSV file header                            | False           |
| `--max-concurrency`    | Maximum concurrent uploads                      | 10              |
| `--retry-count`        | Number of retry attempts for failures           | 3               |
| `--dry-run`            | Simulate without actual uploads                 | False           |
| `--verbose`, `-v`      | Increase verbosity (can be used multiple times) | 0               |
| `--no-progress`        | Disable progress bar                            | False           |

## 💡 Pro Tips & Best Practices

After uploading more files than I care to count, here are some hard-won insights:

1. **Start Small**: Test with a small subset before committing to a 400k file upload

2. **Monitor Bandwidth**: If you're running this in production, be mindful of your network capacity

3. **Appropriate Concurrency**:
    - Low-powered machine: `--max-concurrency 5`
    - Average laptop/desktop: `--max-concurrency 10`
    - Beefy server with great connection: `--max-concurrency 20`

4. **Storage Class Selection**: Use `--storage-class STANDARD_IA` for less frequently accessed assets to save costs

5. **Failure Analysis**: After completion, check logs and failure reports to identify patterns in any failed uploads

6. **Resume Strategy**: If a large upload job fails, you can:
    - Fix the failure report CSV (remove successfully uploaded files)
    - Use it as the new input file for a subsequent run

## 🛠️ Troubleshooting

| Problem                   | Possible Solution                                |
|---------------------------|--------------------------------------------------|
| **Slow uploads**          | Reduce concurrency, check network bandwidth      |
| **Memory issues**         | Reduce concurrency to lower memory usage         |
| **AWS credential errors** | Verify credentials with `aws s3 ls`              |
| **Files not found**       | Verify paths are accessible from script location |
| **Permission denied**     | Check file system permissions on source files    |

## 📊 Performance Analysis

Based on extensive testing, here's what you can expect:

| Network Speed | Files/second* | MB/second* |
|---------------|---------------|------------|
| 100 Mbps      | ~5-10         | ~10        |
| 500 Mbps      | ~15-30        | ~50        |
| 1 Gbps        | ~25-50        | ~100       |

*Depends on file size, concurrency settings, and AWS region proximity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The boto3 team for their excellent AWS SDK
- tqdm for the lovely progress bars
- Everyone who's ever had to upload thousands of files to S3 and felt the pain

---

*"I spent so much time uploading files that I automated myself out of a job. Worth it."* - [The Developer](https://github.com/poacosta)