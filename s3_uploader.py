#!/usr/bin/env python3
"""
S3 Mass Upload Tool - A robust solution for uploading large numbers of files to S3
with error handling, logging, and performance monitoring.
"""
import os
import sys
import time
import csv
import logging
import argparse
import concurrent.futures
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import uuid

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
import tqdm


@dataclass
class UploadStats:
    """Statistics for tracking upload performance."""
    total_files: int = 0
    uploaded_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_bytes: int = 0
    start_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def avg_speed_mbps(self) -> float:
        """Return average upload speed in MB/s."""
        if self.elapsed_time == 0:
            return 0
        return (self.total_bytes / 1024 / 1024) / self.elapsed_time


# pylint: disable=R0903
class S3MassUploader:
    """Handles uploading multiple files to S3 with robust error handling and logging."""

    def __init__(self, config: argparse.Namespace):
        """Initialize the uploader with configuration."""
        self.config = config
        self.stats = UploadStats()
        self.stats.start_time = time.time()

        self._setup_logging()
        self._setup_s3_client()

        self.files_to_upload = self._load_file_list()
        self.stats.total_files = len(self.files_to_upload)

        self.transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=config.max_concurrency,
            multipart_chunksize=8 * 1024 * 1024,
            use_threads=True
        )

        self.failed_uploads: List[Tuple[str, str, Exception]] = []

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity level."""
        log_level = logging.WARNING
        if self.config.verbose == 1:
            log_level = logging.INFO
        elif self.config.verbose >= 2:
            log_level = logging.DEBUG

        if not os.path.exists('logs'):
            os.makedirs('logs')

        log_filename = f"logs/s3_upload_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.log"

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("S3Uploader")
        self.logger.info("Logging initialized. Log file: %s", log_filename)

    def _setup_s3_client(self) -> None:
        """Set up the S3 client with proper configuration."""
        session = boto3.Session(
            profile_name=self.config.profile if hasattr(self.config, 'profile') else None
        )

        self.s3_client = session.client(
            's3',
            region_name=self.config.region
        )

        if not self.config.dry_run:
            try:
                self.s3_client.head_bucket(Bucket=self.config.bucket)
                self.logger.info("Successfully connected to bucket: %s", self.config.bucket)
            except ClientError as e:
                self.logger.error("Failed to access bucket %s: %s", self.config.bucket, str(e))
                sys.exit(1)

    def _load_file_list(self) -> List[str]:
        """Load list of files to upload from CSV or TXT file."""
        file_paths = []

        if not os.path.exists(self.config.file_list):
            self.logger.error("File list not found: %s", self.config.file_list)
            sys.exit(1)

        self.logger.info("Loading file list from: %s", self.config.file_list)

        try:
            if self.config.file_list.lower().endswith('.csv'):
                with open(self.config.file_list, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    if self.config.skip_header:
                        next(reader, None)
                    for row in reader:
                        if row:
                            file_path = row[self.config.path_column]
                            file_paths.append(file_path)
            else:
                with open(self.config.file_list, 'r', encoding='utf-8') as f:
                    file_paths = [line.strip() for line in f if line.strip()]

            self.logger.info("Loaded %d file paths for processing", len(file_paths))
            return file_paths

        except (IOError, csv.Error) as e:
            self.logger.error("Failed to load file list: %s", str(e))
            sys.exit(1)

    def _get_s3_key(self, file_path: str) -> str:
        """
        Generate the S3 key (destination path) for a file preserving the immediate folder structure.
        """
        source_path = Path(file_path)

        if self.config.base_path:
            try:
                rel_path = source_path.relative_to(Path(self.config.base_path))
            except ValueError:
                rel_path = source_path.name
        else:
            rel_path = source_path

        if self.config.destination_prefix:
            s3_key = f"{self.config.destination_prefix.rstrip('/')}/{rel_path}"
        else:
            s3_key = str(rel_path)

        return s3_key.replace('\\', '/')

    def _upload_file(self, file_path: str) -> Tuple[bool, Optional[Exception]]:
        """
        Upload a single file to S3 with error handling.
        Returns (success, exception) tuple.
        """
        if not os.path.exists(file_path):
            self.logger.warning("File does not exist: %s", file_path)
            return False, FileNotFoundError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        s3_key = self._get_s3_key(file_path)

        try:
            self.logger.debug(
                "Uploading %s (%.2f MB) to s3://%s/%s",
                file_path,
                file_size / 1024 / 1024,
                self.config.bucket,
                s3_key
            )

            if self.config.dry_run:
                self.logger.info(
                    "DRY RUN: Would upload %s to s3://%s/%s",
                    file_path,
                    self.config.bucket,
                    s3_key
                )
                time.sleep(0.01)
                return True, None

            self.s3_client.upload_file(
                file_path,
                self.config.bucket,
                s3_key,
                Config=self.transfer_config,
                ExtraArgs={
                    'StorageClass': self.config.storage_class,
                    'ContentType': self._guess_content_type(file_path),
                }
            )

            return True, None

        except (ClientError, IOError) as e:
            self.logger.error("Failed to upload %s: %s", file_path, str(e))
            return False, e

    @staticmethod
    def _guess_content_type(file_path: str) -> str:
        """Guess the MIME type based on file extension."""
        extension = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp',
        }
        return content_types.get(extension, 'application/octet-stream')

    def run(self) -> None:
        """Execute the upload process."""
        self.logger.info("Starting upload process")
        self.logger.info("Total files to process: %d", self.stats.total_files)
        self.logger.info("Mode: %s", 'DRY RUN' if self.config.dry_run else 'UPLOAD')

        progress_bar = tqdm.tqdm(
            total=self.stats.total_files,
            unit="file",
            desc="Uploading",
            disable=self.config.no_progress
        )

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_concurrency
        ) as executor:
            future_to_file = {
                executor.submit(self._upload_file, file_path): file_path
                for file_path in self.files_to_upload
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    success, exception = future.result()
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

                    if success:
                        self.stats.uploaded_files += 1
                        self.stats.total_bytes += file_size
                        self.logger.debug("Successfully uploaded: %s", file_path)
                    else:
                        self.stats.failed_files += 1
                        self.failed_uploads.append(
                            (file_path, self._get_s3_key(file_path), exception)
                        )

                except Exception as e:  # pylint: disable=W0718
                    # Catch all exceptions to ensure robustness of the upload process
                    self.stats.failed_files += 1
                    self.logger.error("Unexpected error processing %s: %s", file_path, str(e))
                    self.failed_uploads.append((file_path, self._get_s3_key(file_path), e))

                progress_bar.update(1)

                if (self.stats.uploaded_files + self.stats.failed_files) % 100 == 0:
                    self._log_progress()

        progress_bar.close()

        if self.config.retry_count > 0 and self.failed_uploads:
            self._retry_failed_uploads()

        self._log_summary()

    def _retry_failed_uploads(self) -> None:
        """Retry failed uploads with exponential backoff."""
        if not self.failed_uploads:
            return

        self.logger.info("Retrying %d failed uploads", len(self.failed_uploads))

        for retry in range(1, self.config.retry_count + 1):
            if not self.failed_uploads:
                break

            self.logger.info(
                "Retry attempt %d/%d for %d files",
                retry,
                self.config.retry_count,
                len(self.failed_uploads)
            )

            current_retries = self.failed_uploads.copy()
            self.failed_uploads = []

            backoff_time = 2 ** retry
            self.logger.info("Waiting %d seconds before retry...", backoff_time)
            time.sleep(backoff_time)

            retry_progress = tqdm.tqdm(
                total=len(current_retries),
                unit="file",
                desc=f"Retry #{retry}",
                disable=self.config.no_progress
            )

            for file_path, s3_key, _ in current_retries:
                try:
                    success, exception = self._upload_file(file_path)

                    if success:
                        self.stats.uploaded_files += 1
                        self.stats.failed_files -= 1
                        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                        self.stats.total_bytes += file_size
                        self.logger.info("Successfully uploaded on retry: %s", file_path)
                    else:
                        self.failed_uploads.append((file_path, s3_key, exception))

                except Exception as e:  # pylint: disable=W0718
                    # Broad exception to ensure retry process continues for all files
                    self.failed_uploads.append((file_path, s3_key, e))
                    self.logger.error("Retry failed for %s: %s", file_path, str(e))

                retry_progress.update(1)

            retry_progress.close()

        if self.failed_uploads:
            self.logger.warning(
                "%d files failed after all retry attempts",
                len(self.failed_uploads)
            )

            failure_report = f"failed_uploads_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            with open(failure_report, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['File Path', 'S3 Key', 'Error Message'])
                for file_path, s3_key, exception in self.failed_uploads:
                    writer.writerow([file_path, s3_key, str(exception)])

            self.logger.info("Failed uploads written to: %s", failure_report)

    def _log_progress(self) -> None:
        """Log current progress statistics."""
        progress_pct = 0
        if self.stats.total_files:
            completed = self.stats.uploaded_files + self.stats.failed_files
            progress_pct = completed / self.stats.total_files * 100

        self.logger.info(
            "Progress: %.1f%% | Uploaded: %d | Failed: %d | "
            "Avg Speed: %.2f MB/s | Elapsed: %s",
            progress_pct,
            self.stats.uploaded_files,
            self.stats.failed_files,
            self.stats.avg_speed_mbps,
            self._format_time(self.stats.elapsed_time)
        )

    def _log_summary(self) -> None:
        """Log the final summary statistics."""
        elapsed = self.stats.elapsed_time

        summary = [
            "===== Upload Summary =====",
            f"Total files processed: {self.stats.total_files}",
            f"Successfully uploaded: {self.stats.uploaded_files}",
            f"Failed uploads: {self.stats.failed_files}",
            f"Skipped files: {self.stats.skipped_files}",
            f"Total data transferred: {self.stats.total_bytes / (1024 * 1024 * 1024):.2f} GB",
            f"Average upload speed: {self.stats.avg_speed_mbps:.2f} MB/s",
            f"Total time: {self._format_time(elapsed)}",
            "=========================="
        ]

        for line in summary:
            self.logger.info("%s", line)

        summary_file = f"upload_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            for line in summary:
                f.write(f"{line}\n")

        self.logger.info("Summary written to: %s", summary_file)

        if self.failed_uploads:
            self.logger.warning(
                "%d files failed. See the failure report for details.",
                len(self.failed_uploads)
            )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload large numbers of files to AWS S3 with error handling and logging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("file_list", help="Path to CSV/TXT file containing list of files to upload")
    parser.add_argument("bucket", help="S3 bucket name")

    parser.add_argument("--base-path", help="Base path to calculate relative paths from")
    parser.add_argument("--destination-prefix", help="Prefix to add to all S3 object keys")
    parser.add_argument("--profile", help="AWS profile name to use")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--storage-class",
        default="STANDARD",
        choices=[
            "STANDARD", "REDUCED_REDUNDANCY", "STANDARD_IA", "ONEZONE_IA",
            "INTELLIGENT_TIERING", "GLACIER", "DEEP_ARCHIVE"
        ],
        help="S3 storage class to use"
    )

    parser.add_argument(
        "--path-column",
        type=int,
        default=0,
        help="Column index (0-based) in CSV containing file paths"
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first line of the CSV file (header)"
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent uploads"
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Number of times to retry failed uploads"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate uploads without actually transferring files"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Don't display progress bar"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    uploader = S3MassUploader(args)
    uploader.run()


if __name__ == "__main__":
    main()
