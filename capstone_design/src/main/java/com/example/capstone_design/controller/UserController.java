package com.example.capstone_design.controller;

import com.example.capstone_design.dto.UserResponse;
import com.example.capstone_design.entity.UserAccount;
import com.example.capstone_design.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {
    private final UserService userService;

    // ✅ 내 정보
    @GetMapping("/me")
    public UserResponse me() {
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        UserAccount u = userService.findByUsername(username);
        return new UserResponse(u.getId(), u.getUsername(), u.getRole(), u.isEnabled(), u.getCreatedAt());
    }

    // ✅ (선택) 관리자만 전체 목록
    @PreAuthorize("hasRole('ADMIN')")
    @GetMapping
    public List<UserResponse> listForAdmin() {
        return userService.list().stream()
                .map(u -> new UserResponse(u.getId(), u.getUsername(), u.getRole(), u.isEnabled(), u.getCreatedAt()))
                .toList();
    }

    // ⛔ 일반 유저용 생성/삭제 엔드포인트는 제거 (회원가입은 /api/auth/signup 사용)
    // 필요하면 관리자 전용으로:
    @PreAuthorize("hasRole('ADMIN')")
    @PostMapping
    public UserResponse adminCreate(@RequestBody com.example.capstone_design.dto.AuthDtos.SignupRequest req) {
        // 관리자가 강제로 유저 생성하려면 AuthService.signup 재사용 or 별도 서비스 작성
        throw new ResponseStatusException(HttpStatus.NOT_IMPLEMENTED, "Use /api/auth/signup or implement admin create");
    }

    @PreAuthorize("hasRole('ADMIN')")
    @DeleteMapping("/{id}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void adminDelete(@PathVariable Long id) {
        userService.delete(id);
    }
}
