import { SignUp } from "@clerk/nextjs";

const SignUpPage = () => {
    return <SignUp afterSignUpUrl="/new-user" fallbackRedirectUrl="/new-user" />
}

export default SignUpPage
