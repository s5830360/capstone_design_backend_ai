package com.example.capstone_design.controller;

import com.example.capstone_design.dto.RecordingResponse;
import com.example.capstone_design.entity.Recording;
import com.example.capstone_design.service.RecordingService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.net.MalformedURLException;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
@RequestMapping("/api/recordings")
@RequiredArgsConstructor
public class RecordingController {
    private final RecordingService recordingService;

    // 파일 업로드 (로그인 필요)
    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @ResponseStatus(HttpStatus.CREATED)
    public RecordingResponse upload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "description", required = false) String description,
            @RequestParam(value = "latitude", required = false) Double latitude,
            @RequestParam(value = "longitude", required = false) Double longitude,
            @RequestParam(value = "address", required = false) String address,
            @RequestParam(value = "emotionPublic", defaultValue = "true") boolean emotionPublic
    ) {
        // 로그인 사용자명 가져오기
        String username = SecurityContextHolder.getContext().getAuthentication().getName();

        Recording saved = recordingService.save(
                username,
                file,
                description,
                latitude,
                longitude,
                address,
                emotionPublic
        );

        return RecordingResponse.from(saved);
    }

    // ===== 파일 다운로드/스트리밍 =====
    @GetMapping("/{id}/file")
    public ResponseEntity<Resource> getFile(@PathVariable Long id) {
        Recording recording = recordingService.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Recording not found"));

        Path filePath = Paths.get(System.getProperty("user.home"), "capstone_uploads", recording.getStoredFilename());

        try {
            Resource resource = new UrlResource(filePath.toUri());
            if (!resource.exists()) {
                throw new ResponseStatusException(HttpStatus.NOT_FOUND, "File not found on server");
            }

            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(recording.getContentType()))
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            "inline; filename=\"" + recording.getOriginalFilename() + "\"")
                    .body(resource);

        } catch (MalformedURLException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "File read error", e);
        }
    }
}
