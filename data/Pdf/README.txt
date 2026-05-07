
Dataset originally created 11/6/2018

- UPDATE 11/19/19 (Corrected digest definition)
- UPDATE 1/3/19 (Renamed image_width, image_height, & content_length to page_width, page_height, & file_size)
- UPDATE 12/27/18 (Added SHA256 and SHA512 checksum fields)
- UPDATE 12/3/18 (Removed offset and filename fields)


I. About This Dataset

This dataset is based on exploratory work begun by the Library of Congress's Web Archiving Team in 2018. The goal of the work was to explore the contents of the corpus of the web archives through analysis of the indexes of the harvested web content, as stored in CDX files (https://web.archive.org/web/20171123000432/https://iipc.github.io/warc-specifications/specifications/cdx-format/cdx-2006/). These indexes were used for initial analysis rather than the archived content, since that content, stored in W/ARC container files (http://webarchive.loc.gov/all/20180105015233/https://www.loc.gov/preservation/digital/formats/fdd/fdd000236.shtml), presents significant challenges due to large size and high processing requirements.

The CDX indexes used in this initial analysis were 6 TB in size, whereas the web archive content in WARC files constituted nearly 1.5 PB at the time of analysis (November 2018).

Most of the information that is provided alongside the files was extracted by the Web Archiving Team, through a process called CDX Line Extraction. See the “How Was It Created” section below for more information on this process.


The filetype dataset was created by filtering the larger set derived from the CDX Line Extraction process, according to the media type pulled from the CDX. The media type information is pulled from CDX field m, which records "mime type" identified in the underlying HTML of a harvested page.

We have referred to this designation by its more current name, "media type" (http://webarchive.loc.gov/all/20171105042213/http://www.iana.org/assignments/media-types/media-types.xhtml), and therefore, all references to "media type" below may be considered to be derived from the MIME type information from the CDX indexes.


II. What's Included?

This dataset includes:

- lcwa_gov_pdf_data.zip - compressed bag containing the 1,000 randomly selected PDFs from the archive, as well as manifest files with sha256 and sha512 checksums. The structure of the content follows the structure of the BagIt specification (see http://webarchive.loc.gov/all/20160830141859/https://tools.ietf.org/html/draft-kunze-bagit-08#section-2).

- lcwa_gov_pdf_metadata.csv - a CSV containing metadata derived from the CDX line
entry for each PDF. The fields and their contents are described in the "dataset
Field Descriptions" section.


III. How Was It Created?

As mentioned above in the “About This Dataset” section, the bulk of this dataset was created using CDX Line Extraction. The extraction process used an Elastic MapReduce (EMR) cluster on AWS cloud services to run a series of MapReduce jobs (see https://en.wikipedia.org/wiki/MapReduce). The jobs filtered and sorted the CDX lines based on the following fields from the CDX line entries:

- digest: a unique cryptographic hash of the web object’s payload at the time of the crawl, which provides a distinct fingerprint for that object; it is a Base32 encoded SHA-1 hash.

- mimetype: two-part designation (type/subtype) that describes the nature and format of the web object,
as reported by the server at the time of the crawl.

- status code: represents the HTTP response code from the server at the time of the crawl, e.g. 200, 404, etc.

- original URL: the URL that was captured during the web harvesting process.

See the CDX specification for more information about the fields: https://web.archive.org/web/20171123000432/https://iipc.github.io/warc-specifications/specifications/cdx-format/cdx-2006/.

The CDX Line Extraction process involved multiple phases. First, a MapReduce job
filtered out lines from the 6TB corpus that were:

1) of the mime type requested, in this case, "application/pdf"

2) had a status code of 200

3) and whose top level domain from the original URL was ".gov"

The query results wrote matching lines to new CDX files, which were stored in an S3 bucket to be used in the next step. In this step, another MapReduce job pulled out all the digests from the filtered CDX files created by the first job and wrote them to a list that was also stored in an S3 bucket. Then, a Python script was used to randomly select one thousand digests from the digest list that was created by the second MapReduce job. Finally, a third MapReduce job took this subsection of one thousand digests as an input and extracted the CDX Line(s) each digest was referenced in, and wrote them to a CDX file that was stored in an S3 bucket. The CDX file from the final MapReduce Job was downloaded, converted to a CSV using a Python script, then used as the basis for any additional metadata extracted from the files or other computational methods of exploration.

Additional information not extracted from the CDX index was also created for use in tracking file integrity and basic metadata. This included the use of the Apache Tika tool to extract metadata from each of the files (see https://tika.apache.org/),
which was then recorded in the accompanying csv. All information derived through Tika is noted below. Response headers information stored with the web archives (content length) furnished the data for the file_size field. If the header did not include the content length, the file size (in bytes) was obtained after the PDF was retrieved. Additionally, a derivative image of the first page from each PDF was created, then run through Tika to ascertain the page width, height, and surface area. Those derivative images are not included in this dataset.


IV. Dataset Field Descriptions

This section lists and describes each of the fields included in lcwa_gov_pdf_metadata.csv. The csv contains 17 fields (listed in the first line), and 1000 lines with the corresponding information for each field as follows:

- urlkey: the url of the captured web object, without the protocol (http://) or the leading www.
This information is extracted from the CDX index file.

- timestamp: timestamp in the form YYYYMMDDhhmmss. The time represents the point
at which the web object was captured, as recorded in the CDX index file.

- original: the url of the captured web object, including the protocol (http://)
and the leading www, if applicable, extracted from the CDX index file.

- mimetype: the mimetype as recorded in the CDX. In this case, all mimetype values
match "application/pdf".

- statuscode: the HTTP response code received from the server at the time of capture, e.g., 200, 404.
In this case, only codes that matched "200" were selected.

- digest: a unique, cryptographic hash function of the web object’s payload at the time of the crawl.
This provides a distinct fingerprint for the object; it is a Base32 encoded SHA-1 hash, derived from the CDX index file.
In this case, the hash was computed with the pdf file as an input.

- pdf_version: the generation of the pdf file, according to information extracted
with Apache Tika.

- creator_tool: the software used to create the pdf, according to information
extracted with Apache Tika. In cases where no information was found, a value of "-"
was recorded.

- producer: the specific library used to generate or encode the pdf document, according
to information extracted with Apache Tika. In cases where no information was found, a value of "-"
was recorded.

- date_created: date of file creation, according to information extracted with
Apache Tika. The date is encoded according to ISO 8601 (YYYY-MM-DDThh:mm:ssZ) and in whatever time the originating
system recorded. In cases where no information was found, a value of "-"
was recorded.

- pages: number of pages in the pdf document, according to information extracted with
Apache Tika.

- page_width: a value that represents the horizontal axis of the document's page size. As a part of the Apache Tika analysis, the first page of each PDF was saved as an image with 72 DPI. The image was then run through Tika to derive width and height in pixels.

- page_height: a value that represents the vertical axis of the document's page size. as a part of the Apache Tika analysis, the first page of each PDF was saved as an image with 72 DPI. The image was then run through Tika to derive width and height in pixels. 

- surface_area: a value that represents the area of the document's page size. As a part of the Apache Tika analysis, the first page of each PDF was saved as an image with 72 DPI. The image was then run through Tika to derive width and height in pixels. This field represents the width divided by the dots per inch (72) times the height divided by the dots per inch, e.g., (612 / 72) * (792 / 72) = 94.

- file_size: the size of the web object, in bytes, derived from additional processing methods used in conjunction with Apache Tika.

- sha256: a unique cryptographic hash of the downloaded web object, computed using the SHA256 function from the SHA-2 algorithm. It serves as a checksum for the downloaded web object and was created during the bagit process.

- sha512: a unique cryptographic hash of the downloaded web object, computed using the SHA512 function from the SHA-2 algorithm. It serves as a checksum for the downloaded web object and was created during the bagit process.


V. Rights Statement

This dataset was derived from content in the Library’s web archives. The Library follows a notification and permission process in the acquisition of content for the web archives, and to allow researcher access to the archived content, as described on the web archiving program page, https://www.loc.gov/programs/web-archiving/about-this-program/. Files were extracted from a variety of archived United States government websites collected in a number of event and thematic archives. See a representative Rights & Access statement for a sample collection which applies to all of the content in this dataset: https://www.loc.gov/collections/legislative-branch-web-archive/about-this-collection/rights-and-access/.


VI. Creator and Contributor Information

Creator: Chase Dooley

Contributors: Jesse Johnston, Aly DesRochers, Grace Thomas, Abbie Grotke


VII. Contact Information

Please direct all questions and comments to webcapture@loc.gov.

